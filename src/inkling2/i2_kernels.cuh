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
/// The attention over the depth, a block per token: the rms-normed states (s0 .. s(S-1)) weighted by a softmax of their products with each of nq
/// queries (scale 1 / sqrt(n)); out[q][t][i].  `t_out`: the token count of `out` (its q stride).
__global__ void __launch_bounds__(kMixThreads)
k_mix(const float* s0, const float* s1, const float* s2, const float* s3, const float* s4, int S, int n, const float* queries, int nq, float* out, int t_out,
      float eps) {
    __shared__ float red[8][kMaxStates * (1 + kMaxQueries)];
    __shared__ float tot[kMaxStates * (1 + kMaxQueries)];
    __shared__ float wgt[kMaxStates][kMaxQueries];
    const float* st[kMaxStates] = {s0, s1, s2, s3, s4};
    const int t = (int) blockIdx.x, tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float ss[kMaxStates], dt[kMaxStates][kMaxQueries];
    for (int s = 0; s < kMaxStates; ++s) {
        ss[s] = 0.f;
        for (int q = 0; q < kMaxQueries; ++q) dt[s][q] = 0.f;
    }
    for (int i = tid; i < n; i += kMixThreads) {
        for (int s = 0; s < S; ++s) {
            const float x = st[s][(size_t) t * (size_t) n + (size_t) i];
            ss[s] += x * x;
            for (int q = 0; q < nq; ++q) dt[s][q] += x * queries[(size_t) q * (size_t) n + (size_t) i];
        }
    }
    for (int s = 0; s < S; ++s) {
        const float a = warp_sum(ss[s]);
        if (lane == 0) red[warp][s * (1 + kMaxQueries)] = a;
        for (int q = 0; q < nq; ++q) {
            const float b = warp_sum(dt[s][q]);
            if (lane == 0) red[warp][s * (1 + kMaxQueries) + 1 + q] = b;
        }
    }
    __syncthreads();
    for (int k = tid; k < S * (1 + kMaxQueries); k += kMixThreads) {
        float a = 0.f;
        for (int w = 0; w < kMixThreads / 32; ++w) a += red[w][k];
        tot[k] = a;
    }
    __syncthreads();
    if (tid < nq) {                               // the softmax of query `tid` over the states (one thread each: S <= 5)
        const int q = tid;
        float sc[kMaxStates], mx = -INFINITY;
        for (int s = 0; s < S; ++s) {
            const float inv = 1.0f / sqrtf(tot[s * (1 + kMaxQueries)] / (float) n + eps);
            sc[s] = tot[s * (1 + kMaxQueries) + 1 + q] * inv * (1.0f / sqrtf((float) n));
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.f;
        for (int s = 0; s < S; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
        for (int s = 0; s < S; ++s) wgt[s][q] = sc[s] / sum * (1.0f / sqrtf(tot[s * (1 + kMaxQueries)] / (float) n + eps));
    }
    __syncthreads();
    for (int i = tid; i < n; i += kMixThreads) {
        for (int q = 0; q < nq; ++q) {
            float o = 0.f;
            for (int s = 0; s < S; ++s) o += wgt[s][q] * st[s][(size_t) t * (size_t) n + (size_t) i];
            out[((size_t) q * (size_t) t_out + (size_t) t) * (size_t) n + (size_t) i] = o;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// The cross-pass controller, a block per token: cur = cur - tanh(forget g) cur + tanh(gain g) m with g = down([rms(cur), rms(m)]).
/// down [rk][2 n], gain / forget [n][rk] (floats); cur is updated in place.
__global__ void __launch_bounds__(kMixThreads)
k_control(float* cur, const float* msg, int n, int rk, const float* down, const float* gain, const float* forget, float eps) {
    __shared__ float sh[32];
    __shared__ float gate[256];
    const int t = (int) blockIdx.x, tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* c = cur + (size_t) t * (size_t) n;
    const float* m = msg + (size_t) t * (size_t) n;
    float a = 0.f, b = 0.f;
    for (int i = tid; i < n; i += kMixThreads) { a += c[i] * c[i]; b += m[i] * m[i]; }
    const float sa = 1.0f / sqrtf(block_sum(a, sh) / (float) n + eps);
    const float sb = 1.0f / sqrtf(block_sum(b, sh) / (float) n + eps);
    for (int r = warp; r < rk; r += kMixThreads / 32) {
        float acc = 0.f;
        const float* d = down + (size_t) r * (size_t) 2 * (size_t) n;
        for (int i = lane; i < n; i += 32) acc += d[i] * (c[i] * sa);
        for (int i = lane; i < n; i += 32) acc += d[n + i] * (m[i] * sb);
        acc = warp_sum(acc);
        if (lane == 0) gate[r] = acc;
    }
    __syncthreads();
    for (int i = tid; i < n; i += kMixThreads) {
        float ga = 0.f, fo = 0.f;
        const float* gr = gain + (size_t) i * (size_t) rk;
        const float* fr = forget + (size_t) i * (size_t) rk;
        for (int r = 0; r < rk; ++r) { ga += gr[r] * gate[r]; fo += fr[r] * gate[r]; }
        c[i] = c[i] - tanhf(fo) * c[i] + tanhf(ga) * m[i];
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
