// i2_kernels.cuh - the Inkling 2 engine's CUDA kernels, beside the Gemma 4 engine's matrix kernels (g4_kernels.cuh: k_mm, k_mm_exp, k_rms,
// k_expert_sum, k_softcap, k_argmax_*), which it reuses.  Written to run unchanged on the CPU emulation (g4_emu.hpp): shared memory is static,
// every thread of a block reaches every __syncthreads / shuffle, blocks are a multiple of 32 threads.
//
// Layout: activations are float, token-major ([T][n]); the K / V cache is half ([kv head][cell][head_dim]); every visit of a layer keeps, besides
// its cache, a ring of the last kRing positions of the four convolution inputs ([position % kRing][channels]: K, V, attention output, FFN output),
// so that a chunk's convolutions read their three previous inputs from the chunk or from the ring.  The math is i2_ref.hpp's.
#pragma once

#include "../gemma4/g4_kernels.cuh"

namespace i2k {

using g4k::block_sum;
using g4k::kThreads;
using g4k::warp_sum;

constexpr int kRing = 128;          // positions of convolution inputs kept per visit (the step back a conversation can take)
constexpr int kMaxExtent = 1024;    // the relative-position bias table of a layer
constexpr int kMixThreads = 256;
constexpr int kMaxStates = 5;       // the residual blocks and the current stream a mix reads
constexpr int kMaxQueries = 4;

G4_DEV float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------------------------------------------------------------------
/// The attention over the depth, in two launches so that a decoded token's 2304 values are not one block's work.
/// k_mix_stats: a block per (state, token), grid (S, T): sum of squares of the state's row and its products with each of the nq queries -> stats[(t * S + s)][1 + kMaxQueries].
__global__ void __launch_bounds__(kMixThreads)
k_mix_stats(const float* s0, const float* s1, const float* s2, const float* s3, const float* s4, int S, int n, const float* queries, int nq, float* stats) {
    __shared__ float red[8][1 + kMaxQueries];
    const int s = (int) blockIdx.x, t = (int) blockIdx.y, tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* x = (s == 0 ? s0 : s == 1 ? s1 : s == 2 ? s2 : s == 3 ? s3 : s4) + (size_t) t * (size_t) n;
    float a = 0.f, d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    for (int i = tid; i < n; i += kMixThreads) {
        const float v = x[i];
        a += v * v;
        d0 += v * queries[i];
        if (nq > 1) d1 += v * queries[(size_t) n + (size_t) i];
        if (nq > 2) d2 += v * queries[(size_t) 2 * (size_t) n + (size_t) i];
        if (nq > 3) d3 += v * queries[(size_t) 3 * (size_t) n + (size_t) i];
    }
    a = warp_sum(a); d0 = warp_sum(d0); d1 = warp_sum(d1); d2 = warp_sum(d2); d3 = warp_sum(d3);
    if (lane == 0) { red[warp][0] = a; red[warp][1] = d0; red[warp][2] = d1; red[warp][3] = d2; red[warp][4] = d3; }
    __syncthreads();
    if (tid < 1 + kMaxQueries) {
        float v = 0.f;
        for (int w = 0; w < kMixThreads / 32; ++w) v += red[w][tid];
        stats[((size_t) t * (size_t) S + (size_t) s) * (1 + kMaxQueries) + (size_t) tid] = v;
    }
}

/// k_mix_apply: grid (ceil(n / 256), T): out[q][t][i] = sum_s w[s][q] state_s[t][i], w = softmax over the states of (state's products with the query) / rms(state) / sqrt(n),
/// times 1 / rms(state).  `t_out`: the token count of `out` (its q stride).
__global__ void __launch_bounds__(kMixThreads)
k_mix_apply(const float* s0, const float* s1, const float* s2, const float* s3, const float* s4, int S, int n, int nq, const float* stats, float* out, int t_out, float eps) {
    const int t = (int) blockIdx.y, i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float* st[kMaxStates] = {s0, s1, s2, s3, s4};
    const float* sp = stats + (size_t) t * (size_t) S * (1 + kMaxQueries);
    float inv[kMaxStates], v[kMaxStates];
#pragma unroll
    for (int s = 0; s < kMaxStates; ++s) {
        inv[s] = s < S ? 1.0f / sqrtf(sp[s * (1 + kMaxQueries)] / (float) n + eps) : 0.f;
        v[s] = s < S ? st[s][(size_t) t * (size_t) n + (size_t) i] : 0.f;
    }
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        if (q >= nq) break;
        float sc[kMaxStates], mx = -INFINITY;
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) {
            sc[s] = s < S ? sp[s * (1 + kMaxQueries) + 1 + q] * inv[s] * (1.0f / sqrtf((float) n)) : -INFINITY;
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.f;
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) { sc[s] = s < S ? expf(sc[s] - mx) : 0.f; sum += sc[s]; }
        float o = 0.f;
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) o += sc[s] / sum * inv[s] * v[s];
        out[((size_t) q * (size_t) t_out + (size_t) t) * (size_t) n + (size_t) i] = o;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// The cross-pass controller: cur = cur - tanh(forget g) cur + tanh(gain g) m with g = down([rms(cur), rms(m)]), down [rk][2 n], gain / forget [n][rk] (floats); cur is
/// updated in place.  Two launches so that its 2.4 MB of weights are not read by one block: k_ctrl_gate (grid (rk / 4, T), 4 warps, a rank row each) writes g[t][rk],
/// k_ctrl_apply (grid (ceil(n / 256), T)) applies it.
__global__ void __launch_bounds__(128)
k_ctrl_gate(const float* cur, const float* msg, int n, int rk, const float* down, float* gate, float eps) {
    __shared__ float sh[32];
    const int t = (int) blockIdx.y, tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* c = cur + (size_t) t * (size_t) n;
    const float* m = msg + (size_t) t * (size_t) n;
    float a = 0.f, b = 0.f;
    for (int i = tid; i < n; i += 128) { a += c[i] * c[i]; b += m[i] * m[i]; }
    const float sa = 1.0f / sqrtf(block_sum(a, sh) / (float) n + eps);
    const float sb = 1.0f / sqrtf(block_sum(b, sh) / (float) n + eps);
    const int r = (int) blockIdx.x * 4 + warp;
    if (r >= rk) return;
    float acc = 0.f;
    const float* d = down + (size_t) r * (size_t) 2 * (size_t) n;
    for (int i = lane; i < n; i += 32) acc += d[i] * (c[i] * sa) + d[n + i] * (m[i] * sb);
    acc = warp_sum(acc);
    if (lane == 0) gate[(size_t) t * (size_t) rk + (size_t) r] = acc;
}

__global__ void __launch_bounds__(kMixThreads)
k_ctrl_apply(float* cur, const float* msg, int n, int rk, const float* gain, const float* forget, const float* gate) {
    __shared__ float g[256];
    const int t = (int) blockIdx.y, tid = (int) threadIdx.x, i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    for (int r = tid; r < rk; r += kMixThreads) g[r] = gate[(size_t) t * (size_t) rk + (size_t) r];
    __syncthreads();
    if (i >= n) return;
    float* c = cur + (size_t) t * (size_t) n;
    const float* m = msg + (size_t) t * (size_t) n;
    float ga = 0.f, fo = 0.f;
    const float* gr = gain + (size_t) i * (size_t) rk;
    const float* fr = forget + (size_t) i * (size_t) rk;
    for (int r = 0; r < rk; ++r) { ga += gr[r] * g[r]; fo += fr[r] * g[r]; }
    c[i] = c[i] - tanhf(fo) * c[i] + tanhf(ga) * m[i];
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Decoded tokens by integer arithmetic (dp4a), for the types whose rows are blocks of small integers: Q8_0 and NVFP4.  The activations are quantized
// once to int8 blocks of 32 (a float scale each, amax / 127) and every row of every matrix that reads them takes 4 products at a time.  This is what llama.cpp's
// mmvq does (the same 0.5 % of noise on the activations); the float kernels above stay for every other type and for reading a prompt.
// ---------------------------------------------------------------------------------------------------------------------------------------
G4_DEV int i2_dp4a(int a, int b, int c) {
#if defined(STRATA_G4_EMU) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610)
    int r = c;
    for (int i = 0; i < 4; ++i) r += (int) (int8_t) ((a >> (8 * i)) & 0xff) * (int) (int8_t) ((b >> (8 * i)) & 0xff);
    return r;
#else
    return __dp4a(a, b, c);
#endif
}
G4_DEV uint32_t i2_perm(uint32_t a, uint32_t b, uint32_t s) {      // __byte_perm without the sign mode: selector nibbles are 0 .. 7
#ifdef STRATA_G4_EMU
    uint32_t r = 0;
    const uint64_t v = (uint64_t) a | ((uint64_t) b << 32);
    for (int i = 0; i < 4; ++i) r |= (uint32_t) ((v >> (8 * ((s >> (4 * i)) & 7))) & 0xffu) << (8 * i);
    return r;
#else
    return __byte_perm(a, b, s);
#endif
}
G4_DEV int i2_ld16(const uint8_t* p) { return (int) (uint32_t) *reinterpret_cast<const uint16_t*>(p); }
G4_DEV int i2_ld32_b2(const uint8_t* p) { return i2_ld16(p) | (i2_ld16(p + 2) << 16); }          // an int from a 2-byte aligned address

/// The 8 nibbles of q4 (four bytes) through a 16-entry int8 table held in four registers: x = the low nibbles' values (bytes 0 .. 3), y = the high nibbles'.
G4_DEV void i2_table16(uint32_t q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3, int& x, int& y) {
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sel = (q4 >> (16 * i)) & 0xffffu;
        const uint32_t lo = i2_perm(t0, t1, sel & 0x7777u);
        const uint32_t hi = i2_perm(t2, t3, sel & 0x7777u);
        tmp[i] = i2_perm(lo, hi, 0x3210u | ((sel & 0x8888u) >> 1));
    }
    x = (int) i2_perm(tmp[0], tmp[1], 0x6420u);
    y = (int) i2_perm(tmp[0], tmp[1], 0x7531u);
}

/// x[row][n] -> xq (int8) and xd (the scale of each block of 32): a warp a block, grid (ceil(n / 256), rows), 256 threads.
__global__ void __launch_bounds__(256)
k_quant_x(const float* x, int n, int8_t* xq, float* xd) {
    const int row = (int) blockIdx.y, warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const int b = (int) blockIdx.x * 8 + warp;
    if (b * 32 >= n) return;
    const float v = x[(size_t) row * (size_t) n + (size_t) b * 32 + (size_t) lane];
    const float amax = g4k::warp_max(fabsf(v));
    const float d = amax / 127.f;
    int q = d > 0.f ? (int) rintf(v / d) : 0;
    q = max(-127, min(127, q));
    xq[(size_t) row * (size_t) n + (size_t) b * 32 + (size_t) lane] = (int8_t) q;
    if (lane == 0) xd[(size_t) row * (size_t) (n / 32) + (size_t) b] = d;
}

/// y[entry * ldy + y_off + r] = (row r of expert grp_exp[entry]) . (entry's activations), a warp a row, 8 warps a block: grid (ceil(n_out / 8), entries).
/// The activations of an entry are xq + entry * x_stride (x_stride 0: every entry reads the same ones); a matrix that is not an expert's: entries 1, grp_exp null.
/// T = g4::T_Q8_0: a lane takes 8 values of a block of 32 (4 lanes a block); T = g4::T_NVFP4: a lane a sub-block of 16 (4 lanes a block of 64).
template <int T>
__global__ void __launch_bounds__(256)
k_mv(const uint8_t* __restrict__ w, long long expert_bytes, long long row_bytes, int n_in, int n_out, const int8_t* __restrict__ xq, const float* __restrict__ xd, long long x_stride,
     const int* __restrict__ grp_exp, float* __restrict__ y, int ldy, int y_off) {
    // the entry's activations, once per block into shared memory (every row of the block reads all of them: from the L2, with no L1 on Pascal, that was twice the weights' traffic)
    __shared__ int xs_i[1024];                                  // up to 4096 int8 values
    __shared__ float ds[128];
    const int entry = (int) blockIdx.y;
    {
        const int* xg = reinterpret_cast<const int*>(xq + (size_t) entry * (size_t) x_stride);
        for (int i = (int) threadIdx.x; i < n_in / 4; i += 256) xs_i[i] = xg[i];
        const float* dg = xd + (size_t) entry * (size_t) (x_stride / 32);
        for (int i = (int) threadIdx.x; i < n_in / 32; i += 256) ds[i] = dg[i];
    }
    __syncthreads();
    const int row = (int) blockIdx.x * 8 + (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (row >= n_out) return;
    const int e = grp_exp ? grp_exp[entry] : 0;
    const uint8_t* rp = w + (size_t) e * (size_t) expert_bytes + (size_t) row * (size_t) row_bytes;
    const int8_t* xe = reinterpret_cast<const int8_t*>(xs_i);
    const float* de = ds;
    float acc = 0.f;
    if constexpr (T == g4::T_Q8_0) {
        const int nb = n_in / 32, part = lane & 3;
        for (int b = lane >> 2; b < nb; b += 8) {
            const uint8_t* blk = rp + (size_t) b * 34;
            const int* xv = reinterpret_cast<const int*>(xe + (size_t) b * 32 + (size_t) part * 8);
            int s = i2_dp4a(i2_ld32_b2(blk + 2 + part * 8), xv[0], 0);
            s = i2_dp4a(i2_ld32_b2(blk + 2 + part * 8 + 4), xv[1], s);
            acc += (float) s * g4::half_to_float((uint16_t) i2_ld16(blk)) * de[b];
        }
    } else {
        const int nb = n_in / 64, sub = lane & 3;
        for (int b = lane >> 2; b < nb; b += 8) {
            const uint8_t* blk = rp + (size_t) b * 36;
            const uint32_t w0 = *reinterpret_cast<const uint32_t*>(blk + 4 + sub * 8), w1 = *reinterpret_cast<const uint32_t*>(blk + 8 + sub * 8);
            const int* xv = reinterpret_cast<const int*>(xe + (size_t) b * 64 + (size_t) sub * 16);
            int a0, a1, b0, b1;
            i2_table16(w0, 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu, a0, a1);      // the E2M1 values, doubled: 0 1 2 3 4 6 8 12, then their negatives
            i2_table16(w1, 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu, b0, b1);
            int s = i2_dp4a(a0, xv[0], 0);                      // elements 0 .. 3
            s = i2_dp4a(b0, xv[1], s);                          // 4 .. 7
            s = i2_dp4a(a1, xv[2], s);                          // 8 .. 11
            s = i2_dp4a(b1, xv[3], s);                          // 12 .. 15
            acc += (float) s * g4::ue4m3_half((int) blk[sub]) * de[b * 2 + (sub >> 1)];
        }
    }
    acc = g4k::warp_sum(acc);
    if (lane == 0) y[(size_t) entry * (size_t) ldy + (size_t) y_off + (size_t) row] = acc;
}

/// Reading a prompt by the same integer arithmetic: NT tokens (or (token, expert) pairs of one expert) a block, a warp two rows, the weights of both decoded once per slice
/// and used for all NT tokens (the activations of a token read from shared memory once for both rows).  The activations of the group are staged in slices of 512 values.
/// Grid (ceil(n_out / 16), groups), 256 threads.
///   dense matrix (grp_exp null): group g = tokens g * NT .. of ntok, rows of x and y the token's own;
///   experts: group g of grp_exp[g] / grp_first[g] / grp_cnt[g] (the tables mm_exp reads): entry i reads row ent_in[first + i] of the activations and writes row
///   ent_out[first + i] of y.
template <int T, int NT>
__global__ void __launch_bounds__(256)
k_mmq(const uint8_t* __restrict__ w, long long expert_bytes, long long row_bytes, int n_in, int n_out, const int8_t* __restrict__ xq, const float* __restrict__ xd,
      const int* __restrict__ grp_exp, const int* __restrict__ grp_first, const int* __restrict__ grp_cnt, const int* __restrict__ ent_in, const int* __restrict__ ent_out,
      int ntok, float* __restrict__ y, int ldy, int y_off) {
    constexpr int KS = 512;
    __shared__ int xs_i[NT * KS / 4];
    __shared__ float ds[NT * (KS / 32)];
    __shared__ int rows_in[NT];
    const int g = (int) blockIdx.y, tid = (int) threadIdx.x, lane = tid & 31;
    int e = 0, first, cnt;
    if (grp_exp) { e = grp_exp[g]; first = grp_first[g]; cnt = grp_cnt[g]; }
    else { first = g * NT; cnt = min(NT, ntok - first); }
    if (tid < NT) rows_in[tid] = tid < cnt ? (ent_in ? ent_in[first + tid] : first + tid) : 0;
    __syncthreads();
    const int rowA = ((int) blockIdx.x * 8 + (tid >> 5)) * 2, rowB = rowA + 1;
    const uint8_t* wbase = w + (size_t) e * (size_t) expert_bytes;
    const uint8_t* rpA = wbase + (size_t) (rowA < n_out ? rowA : 0) * (size_t) row_bytes;
    const uint8_t* rpB = wbase + (size_t) (rowB < n_out ? rowB : (rowA < n_out ? rowA : 0)) * (size_t) row_bytes;
    float accA[NT], accB[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) { accA[i] = 0.f; accB[i] = 0.f; }
    for (int k0 = 0; k0 < n_in; k0 += KS) {
        const int kn = min(KS, n_in - k0);
        for (int idx = tid; idx < cnt * (kn / 4); idx += 256) {
            const int i = idx / (kn / 4), j = idx % (kn / 4);
            xs_i[i * (KS / 4) + j] = reinterpret_cast<const int*>(xq + (size_t) rows_in[i] * (size_t) n_in + (size_t) k0)[j];
        }
        for (int idx = tid; idx < cnt * (kn / 32); idx += 256) {
            const int i = idx / (kn / 32), j = idx % (kn / 32);
            ds[i * (KS / 32) + j] = xd[(size_t) rows_in[i] * (size_t) (n_in / 32) + (size_t) (k0 / 32) + (size_t) j];
        }
        __syncthreads();
        if constexpr (T == g4::T_Q8_0) {
            const int nb = kn / 32, part = lane & 3;
            for (int b = lane >> 2; b < nb; b += 8) {
                const uint8_t* ba = rpA + (size_t) (k0 / 32 + b) * 34;
                const uint8_t* bb = rpB + (size_t) (k0 / 32 + b) * 34;
                const int a0 = i2_ld32_b2(ba + 2 + part * 8), a1 = i2_ld32_b2(ba + 2 + part * 8 + 4);
                const int c0 = i2_ld32_b2(bb + 2 + part * 8), c1 = i2_ld32_b2(bb + 2 + part * 8 + 4);
                const float da = g4::half_to_float((uint16_t) i2_ld16(ba)), db = g4::half_to_float((uint16_t) i2_ld16(bb));
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    if (i < cnt) {
                        const int* xv = xs_i + i * (KS / 4) + b * 8 + part * 2;
                        const int x0 = xv[0], x1 = xv[1];
                        const float dx = ds[i * (KS / 32) + b];
                        accA[i] += (float) i2_dp4a(a1, x1, i2_dp4a(a0, x0, 0)) * da * dx;
                        accB[i] += (float) i2_dp4a(c1, x1, i2_dp4a(c0, x0, 0)) * db * dx;
                    }
                }
            }
        } else {
            const int nb = kn / 64, sub = lane & 3;
            for (int b = lane >> 2; b < nb; b += 8) {
                const uint8_t* ba = rpA + (size_t) (k0 / 64 + b) * 36;
                const uint8_t* bb = rpB + (size_t) (k0 / 64 + b) * 36;
                int a0, a1, b0, b1, c0, c1, d0, d1;
                i2_table16(*reinterpret_cast<const uint32_t*>(ba + 4 + sub * 8), 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu, a0, a1);
                i2_table16(*reinterpret_cast<const uint32_t*>(ba + 8 + sub * 8), 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu, b0, b1);
                i2_table16(*reinterpret_cast<const uint32_t*>(bb + 4 + sub * 8), 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu, c0, c1);
                i2_table16(*reinterpret_cast<const uint32_t*>(bb + 8 + sub * 8), 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu, d0, d1);
                const float da = g4::ue4m3_half((int) ba[sub]), db = g4::ue4m3_half((int) bb[sub]);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    if (i < cnt) {
                        const int* xv = xs_i + i * (KS / 4) + b * 16 + sub * 4;
                        const int x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
                        const float dx = ds[i * (KS / 32) + b * 2 + (sub >> 1)];
                        int sa = i2_dp4a(a0, x0, 0);
                        sa = i2_dp4a(b0, x1, sa);
                        sa = i2_dp4a(a1, x2, sa);
                        sa = i2_dp4a(b1, x3, sa);
                        int sb = i2_dp4a(c0, x0, 0);
                        sb = i2_dp4a(d0, x1, sb);
                        sb = i2_dp4a(c1, x2, sb);
                        sb = i2_dp4a(d1, x3, sb);
                        accA[i] += (float) sa * da * dx;
                        accB[i] += (float) sb * db * dx;
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (i < cnt) {
            const float va = g4k::warp_sum(accA[i]), vb = g4k::warp_sum(accB[i]);
            if (lane == 0) {
                float* yr = y + (size_t) (ent_out ? ent_out[first + i] : first + i) * (size_t) ldy + (size_t) y_off;
                if (rowA < n_out) yr[rowA] = va;
                if (rowB < n_out) yr[rowB] = vb;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// The raw convolution input `back` positions before token t of the chunk (back = 0: the token itself): from the chunk's own buffer, else from the
/// ring of the visit, else zero before the first position.
G4_DEV float conv_in(const float* buf, int ld, int c, int t, int back, const float* ring, int ch, int seg_off, int pos0) {
    const int idx = t - back;
    if (idx >= 0) return buf[(size_t) idx * (size_t) ld + (size_t) c];
    const int p = pos0 + idx;
    return p >= 0 ? ring[(size_t) (p % kRing) * (size_t) ch + (size_t) seg_off + (size_t) c] : 0.f;
}

/// The head-wise preparation of the projections of T tokens, a block (128 threads) per (token, part): grid (T, n_head + 2 n_kv + n_head).
///   parts [0, n_head): a query head: * q_clip, clip(qk_clip), * tau on a global layer, written back over q
///   [n_head, n_head + n_kv): a K head: convolution (with the input taps on a global layer), clip, half into the cache at cell pos % ring
///   [.., + n_kv): a V head: convolution, rms (no weight), half into the cache
///   the rest: a relative-position head: * rel_clip, clip(rel_clip_thr), * tau on a global layer, written back over r
/// tau = 1 + alpha log(max(1, (pos + 1) / floor)).
__global__ void __launch_bounds__(kThreads)
k_prep(float* q, const float* kraw, const float* vraw, float* r, const float* ktap, const float* vtap, const float* kconv, const float* vconv,
       const float* q_clip, const float* rel_clip, const float* cring, int ch, uint16_t* kc, uint16_t* vc, uint8_t* kq, uint8_t* vq, uint16_t* ks, uint16_t* vs, int kvq,
       int n_head, int n_kv, int hd, int n_rel, int n_conv,
       int pos0, int ring, bool global, float qk_clip, float rel_thr, float eps, float alpha, float floor_) {
    __shared__ float sh[32];
    const int t = (int) blockIdx.x, part = (int) blockIdx.y, i = (int) threadIdx.x;
    const int pos = pos0 + t;
    const int qd = n_head * hd, kd = n_kv * hd;
    const float tau = global ? 1.0f + alpha * logf(fmaxf(1.0f, (float) (pos + 1) / floor_)) : 1.0f;
    if (part < n_head) {
        const int h = part;
        const float v = i < hd ? q[(size_t) t * (size_t) qd + (size_t) h * (size_t) hd + (size_t) i] * q_clip[h * hd + i] : 0.f;
        const float norm = sqrtf(block_sum(v * v, sh)) + 1e-6f;
        const float f = fminf(1.0f, fmaxf(0.0f, qk_clip / norm));
        if (i < hd) q[(size_t) t * (size_t) qd + (size_t) h * (size_t) hd + (size_t) i] = global ? v * f * tau : v * f;
        return;
    }
    if (part < n_head + 2 * n_kv) {
        const bool is_k = part < n_head + n_kv;
        const int kvh = is_k ? part - n_head : part - n_head - n_kv;
        const float* raw = is_k ? kraw : vraw;
        const float* w = is_k ? kconv : vconv;
        const float* tap = global ? (is_k ? ktap : vtap) : nullptr;
        const int seg = is_k ? 0 : kd;
        float x = 0.f;
        if (i < hd) {
            const int c = kvh * hd + i;
            float acc = 0.f;
            for (int k = 0; k < n_conv; ++k) acc += w[(size_t) c * (size_t) n_conv + (size_t) k] * conv_in(raw, kd, c, t, n_conv - 1 - k, cring, ch, seg, pos0);
            if (tap)
                for (int lag = 0; lag < n_conv; ++lag)
                    acc += tap[(size_t) t * (size_t) (n_conv * n_kv) + (size_t) kvh * (size_t) n_conv + (size_t) lag] * conv_in(raw, kd, c, t, lag, cring, ch, seg, pos0);
            x = (raw[(size_t) t * (size_t) kd + (size_t) c] + acc) / sqrtf((float) ((tap ? 2 : 1) * n_conv + 1));
        }
        float y;
        if (is_k) {
            const float norm = sqrtf(block_sum(x * x, sh)) + 1e-6f;
            y = x * fminf(1.0f, fmaxf(0.0f, qk_clip / norm));
        } else {
            y = x * (1.0f / sqrtf(block_sum(x * x, sh) / (float) hd + eps));
        }
        const size_t cellno = (size_t) kvh * (size_t) ring + (size_t) (pos % ring);
        if (kvq == 0) {
            if (i < hd) (is_k ? kc : vc)[cellno * (size_t) hd + (size_t) i] = g4::float_to_half(y);
        } else if ((i >> 5) * 32 < hd) {                   // quantized K / V: blocks of 32 values (a warp), a half scale each: q8_0 (scale amax / 127) or 4 bits (amax / 7)
            const int lane = i & 31, blk = i >> 5;
            const float qmax = kvq == 1 ? 127.f : 7.f;
            const uint16_t dh = g4::float_to_half(g4k::warp_max(fabsf(y)) / qmax);
            const float d = g4::half_to_float(dh);
            int qi = d > 0.f ? (int) rintf(y / d) : 0;
            if (kvq == 1) {
                qi = max(-127, min(127, qi));
                (is_k ? kq : vq)[cellno * (size_t) hd + (size_t) i] = (uint8_t) (int8_t) qi;
            } else {
                qi = max(-8, min(7, qi)) + 8;
                const int other = __shfl_xor_sync(0xffffffffu, qi, 16);
                if (lane < 16) (is_k ? kq : vq)[cellno * (size_t) (hd / 2) + (size_t) (blk * 16 + lane)] = (uint8_t) (qi | (other << 4));
            }
            if (lane == 0) (is_k ? ks : vs)[cellno * (size_t) (hd / 32) + (size_t) blk] = dh;
        }
        return;
    }
    const int h = part - n_head - 2 * n_kv;
    const float v = i < n_rel ? r[(size_t) t * (size_t) (n_head * n_rel) + (size_t) h * (size_t) n_rel + (size_t) i] * rel_clip[h * n_rel + i] : 0.f;
    const float norm = sqrtf(block_sum(v * v, sh)) + 1e-6f;
    const float f = fminf(1.0f, fmaxf(0.0f, rel_thr / norm));
    if (i < n_rel) r[(size_t) t * (size_t) (n_head * n_rel) + (size_t) h * (size_t) n_rel + (size_t) i] = global ? v * f * tau : v * f;
}

/// ring[(pos0 + t) % kRing][seg_off + c] = buf[t][c] for the last kRing tokens of the chunk: after the convolutions that read the ring, never in them
__global__ void __launch_bounds__(kThreads)
k_ring_store(const float* buf, int ld, int n, float* ring, int ch, int seg_off, int pos0, int T) {
    const int t = (int) blockIdx.y, c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= n || t < T - kRing) return;
    ring[(size_t) ((pos0 + t) % kRing) * (size_t) ch + (size_t) seg_off + (size_t) c] = buf[(size_t) t * (size_t) ld + (size_t) c];
}

/// out[t][c] = resid[t][c] + (in[t][c] + sum_k w[c][k] x_{t-3+k}[c]) / sqrt(n_conv + 1)   (resid may be null: just the convolution)
__global__ void __launch_bounds__(kThreads)
k_conv_res(const float* in, int n, const float* w, int n_conv, const float* cring, int ch, int seg_off, int pos0, const float* resid, float* out) {
    const int t = (int) blockIdx.y, c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= n) return;
    float acc = 0.f;
    for (int k = 0; k < n_conv; ++k) acc += w[(size_t) c * (size_t) n_conv + (size_t) k] * conv_in(in, n, c, t, n_conv - 1 - k, cring, ch, seg_off, pos0);
    const float v = (in[(size_t) t * (size_t) n + (size_t) c] + acc) / sqrtf((float) (n_conv + 1));
    out[(size_t) t * (size_t) n + (size_t) c] = resid ? resid[(size_t) t * (size_t) n + (size_t) c] + v : v;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// One element of a cached K / V vector: lane + 32 s of the cell's head vector, in f16 (KVQ 0), q8_0 (1) or 4 bits (2) with a half scale per 32 values.
template <int KVQ, int NU>
__device__ __forceinline__ float kv_at(const uint16_t* h, const uint8_t* qv, const uint16_t* sc, size_t cell, int s, int lane) {
    if constexpr (KVQ == 0) {
        return g4::half_to_float(h[cell * (size_t) (NU * 32) + (size_t) (lane + 32 * s)]);
    } else if constexpr (KVQ == 1) {
        return (float) (int) (int8_t) qv[cell * (size_t) (NU * 32) + (size_t) (lane + 32 * s)] * g4::half_to_float(sc[cell * (size_t) NU + (size_t) s]);
    } else {
        const int b = qv[cell * (size_t) (NU * 16) + (size_t) (s * 16 + (lane & 15))];
        return (float) ((lane < 16 ? (b & 0xF) : (b >> 4)) - 8) * g4::half_to_float(sc[cell * (size_t) NU + (size_t) s]);
    }
}

/// Attention of the T tokens at positions pos0 .. pos0 + T - 1, a block per (token, query head): grid (T, n_head), 4 warps.  Each warp takes every
/// fourth key of the token's window (the last `window` cells on a local layer, every cell on a global one), keeps an online softmax and the weighted sum
/// of the values; the sink is a first key of logit 0 and value 0 (warp 0 starts with it).  The relative-position bias of a key at distance d < extent
/// comes from a table of the block's, a key further away has the table's last, zero row: -sink only.  The caches hold halves, q8_0 or 4-bit values.
/// With nsplit > 1 (one token, a long window) the keys are cut into nsplit parts, a block (grid z) each, whose (max, sum, weighted values) go to pm / pl /
/// pacc for k_attn_merge; the sink belongs to part 0.
template <int NU, int KVQ>
__global__ void __launch_bounds__(kThreads)
k_attn(const float* q, const float* r, const float* rel_bias, const float* sink, const float* gate, const uint16_t* kc, const uint16_t* vc, const uint8_t* kq, const uint8_t* vq,
       const uint16_t* ks, const uint16_t* vs, float* out, int n_head, int n_kv, int n_rel, int extent, int pos0, int ring, int window, int nsplit, float* pm, float* pl, float* pacc) {
    constexpr int HD = NU * 32;
    __shared__ float tbl[kMaxExtent];
    __shared__ float wm[4], wl[4];
    __shared__ float wacc[4][HD];
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t = (int) blockIdx.x, hh = (int) blockIdx.y, sp = (int) blockIdx.z;
    const int kk = hh / (n_head / n_kv);
    const int p = pos0 + t;
    const int lo = window > 0 ? max(0, p - window + 1) : 0;
    const float* rh = r + (size_t) t * (size_t) (n_head * n_rel) + (size_t) hh * (size_t) n_rel;
    const float sk = sink[(size_t) t * (size_t) n_head + (size_t) hh];
    const int n_e = min(extent, p - lo + 1);
    for (int e = tid; e < n_e; e += (int) blockDim.x) {
        float b = 0.f;
        for (int k = 0; k < n_rel; ++k) b += rel_bias[(size_t) e * (size_t) n_rel + (size_t) k] * rh[k];
        tbl[e] = b - sk;
    }
    __syncthreads();
    const float* qh = q + (size_t) t * (size_t) (n_head * HD) + (size_t) hh * (size_t) HD;
    float qv[NU], acc[NU];
#pragma unroll
    for (int s = 0; s < NU; ++s) { qv[s] = qh[lane + 32 * s]; acc[s] = 0.f; }
    const float scale = 1.0f / sqrtf((float) HD);
    const bool first = sp == 0;                                         // the sink is part 0's
    float m = (warp == 0 && first) ? 0.f : -INFINITY, l = (warp == 0 && first) ? 1.f : 0.f;
    const size_t cells = (size_t) kk * (size_t) ring;
    const uint16_t* kh = kc ? kc + cells * (size_t) HD : nullptr;
    const uint16_t* vh = vc ? vc + cells * (size_t) HD : nullptr;
    const uint8_t* kqb = kq ? kq + cells * (size_t) (KVQ == 2 ? HD / 2 : HD) : nullptr;
    const uint8_t* vqb = vq ? vq + cells * (size_t) (KVQ == 2 ? HD / 2 : HD) : nullptr;
    const uint16_t* ksb = ks ? ks + cells * (size_t) NU : nullptr;
    const uint16_t* vsb = vs ? vs + cells * (size_t) NU : nullptr;
    int j0 = lo, j1 = p;
    if (nsplit > 1) {
        const int per = (p - lo + 1 + nsplit - 1) / nsplit;
        j0 = lo + sp * per;
        j1 = min(p, j0 + per - 1);
    }
    for (int j = j0 + warp; j <= j1; j += 4) {
        const size_t cell = (size_t) (j % ring);
        float sc = 0.f;
#pragma unroll
        for (int s = 0; s < NU; ++s) sc += qv[s] * kv_at<KVQ, NU>(kh, kqb, ksb, cell, s, lane);
        sc = warp_sum(sc);
        const int dist = p - j;
        sc = sc * scale + (dist < extent ? tbl[dist] : -sk);
        const float mn = fmaxf(m, sc);
        const float a = m == -INFINITY ? 0.f : expf(m - mn);
        const float b = expf(sc - mn);
        l = l * a + b;
#pragma unroll
        for (int s = 0; s < NU; ++s) acc[s] = acc[s] * a + b * kv_at<KVQ, NU>(vh, vqb, vsb, cell, s, lane);
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
#pragma unroll
    for (int s = 0; s < NU; ++s) wacc[warp][lane + 32 * s] = acc[s];
    __syncthreads();
    float M = wm[0];
    for (int w = 1; w < 4; ++w) M = fmaxf(M, wm[w]);
    float L = 0.f;
    for (int w = 0; w < 4; ++w) L += wm[w] == -INFINITY ? 0.f : wl[w] * expf(wm[w] - M);
    if (nsplit > 1) {                                                   // a part: its state, not yet divided by the sum
        const size_t pi = (size_t) hh * (size_t) nsplit + (size_t) sp;
        if (tid == 0) { pm[pi] = M; pl[pi] = L; }
        for (int e = tid; e < HD; e += (int) blockDim.x) {
            float a = 0.f;
            for (int w = 0; w < 4; ++w) a += wm[w] == -INFINITY ? 0.f : wacc[w][e] * expf(wm[w] - M);
            pacc[pi * (size_t) HD + (size_t) e] = a;
        }
        return;
    }
    const float inv = 1.0f / L;
    float* o = out + (size_t) t * (size_t) (n_head * HD) + (size_t) hh * (size_t) HD;
    const float* gt = gate + (size_t) t * (size_t) (n_head * HD) + (size_t) hh * (size_t) HD;
    for (int e = tid; e < HD; e += (int) blockDim.x) {
        float a = 0.f;
        for (int w = 0; w < 4; ++w) a += wm[w] == -INFINITY ? 0.f : wacc[w][e] * expf(wm[w] - M);
        o[e] = a * inv * sigmoidf(gt[e]);
    }
}

/// The parts of one token's attention put together: per query head, the parts' states merged (grid n_head, HD threads), times the output gate.
template <int NU>
__global__ void __launch_bounds__(kThreads)
k_attn_merge(const float* gate, const float* pm, const float* pl, const float* pacc, float* out, int nsplit) {
    constexpr int HD = NU * 32;
    const int hh = (int) blockIdx.x, e = (int) threadIdx.x;
    if (e >= HD) return;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pm[(size_t) hh * (size_t) nsplit + (size_t) s]);
    float L = 0.f, a = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const size_t pi = (size_t) hh * (size_t) nsplit + (size_t) s;
        if (pm[pi] == -INFINITY) continue;
        const float w = expf(pm[pi] - M);
        L += pl[pi] * w;
        a += pacc[pi * (size_t) HD + (size_t) e] * w;
    }
    out[(size_t) hh * (size_t) HD + (size_t) e] = a / L * sigmoidf(gate[(size_t) hh * (size_t) HD + (size_t) e]);
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// out[r][i] = situ(gate[r][i], up[r][i]) for gu rows [gate | up] of 2 n floats; situ(g, u) = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb).
/// The rows r = rows[blockIdx.y] (or blockIdx.y when rows is null).
__global__ void __launch_bounds__(kThreads)
k_situ(const float* gu, float* act, int n, float beta, float lb, const int* rows, const float* gscale, const int* topk_idx) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const size_t r = rows ? (size_t) rows[blockIdx.y] : (size_t) blockIdx.y;
    const float sc = gscale ? gscale[topk_idx[r]] : 1.f;       // the expert's scalar on the gate+up product (NVFP4 files)
    const float g = sc * gu[r * (size_t) 2 * (size_t) n + (size_t) i], u = sc * gu[r * (size_t) 2 * (size_t) n + (size_t) n + (size_t) i];
    const float a = beta * tanhf(g / beta) * sigmoidf(g);
    act[r * (size_t) n + (size_t) i] = a * (lb * tanhf(u / lb));
}

/// The router's selection, a block of 128 threads per token: p = sigmoid(logit), the n_used best of p + bias (the lowest index among equals), their
/// weights p / (sum + 1e-20) * scale.  logits [T][n_expert]; idx / w [T][n_used].
__global__ void __launch_bounds__(kThreads)
k_topk(const float* logits, const float* bias, int n_expert, int n_used, float scale, int* idx, float* w) {
    __shared__ float p[512], sel[512];
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    __shared__ int chosen[32];
    const int t = (int) blockIdx.x, tid = (int) threadIdx.x;
    for (int e = tid; e < n_expert; e += (int) blockDim.x) {
        p[e] = sigmoidf(logits[(size_t) t * (size_t) n_expert + (size_t) e]);
        sel[e] = p[e] + (bias ? bias[e] : 0.f);
    }
    __syncthreads();
    for (int k = 0; k < n_used; ++k) {
        float v = -INFINITY;
        int ix = 0x7fffffff;
        for (int e = tid; e < n_expert; e += (int) blockDim.x)
            if (sel[e] > v) { v = sel[e]; ix = e; }
        bv[tid] = v;
        bi[tid] = ix;
        __syncthreads();
        for (int s = (int) blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && (bv[tid + s] > bv[tid] || (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]))) { bv[tid] = bv[tid + s]; bi[tid] = bi[tid + s]; }
            __syncthreads();
        }
        if (tid == 0) { chosen[k] = bi[0]; sel[bi[0]] = -INFINITY; }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.f;
        for (int k = 0; k < n_used; ++k) sum += p[chosen[k]];
        sum += 1e-20f;
        for (int k = 0; k < n_used; ++k) {
            idx[(size_t) t * (size_t) n_used + (size_t) k] = chosen[k];
            w[(size_t) t * (size_t) n_used + (size_t) k] = p[chosen[k]] / sum * scale;
        }
    }
}

}  // namespace i2k
