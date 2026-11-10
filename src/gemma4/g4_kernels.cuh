// g4_kernels.cuh - the Gemma 4 engine's CUDA kernels.  Written to run unchanged on the CPU emulation (g4_emu.hpp): shared memory is declared
// static, every thread of a block reaches every __syncthreads / shuffle, and blocks are a multiple of 32 threads.
//
// Layout conventions: activations are float, token-major ([T][n]); the K / V cache is IEEE half stored as uint16_t, [kv head][cell][head_dim];
// a weight matrix is `n_out` rows of `n_in` values in ggml blocks (g4_quant.hpp), an expert tensor `n_expert` of them back to back.
#pragma once

#include "g4_cuda_compat.hpp"
#include "g4_quant.hpp"

namespace g4k {

constexpr int kWarp = 32;
constexpr int kMmWarps = 8;                       // output rows per block of the matrix kernels (a warp each)
constexpr int kMmThreads = kMmWarps * kWarp;
constexpr int kMmTile = 512;                      // values of the activations staged in shared memory at a time
constexpr int kThreads = 128;                     // the other kernels
constexpr int kMaxExperts = 512;
constexpr int kMaxHead = 512;

G4_DEV float warp_sum(float v) {
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
G4_DEV float warp_max(float v) {
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}
/// the sum of v over the block, to every thread; `sh` is a shared array of 32 floats
G4_DEV float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int) blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < nw; ++k) t += sh[k];
    __syncthreads();
    return t;
}
G4_DEV float gelu(float x) {                      // gelu_pytorch_tanh, as ggml_gelu_f32
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x)));
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// y[t][y_off + row] = row of W . x[t], for the tokens t0 .. t0 + NT of a tile (blockIdx.y), W of type T.  A warp takes a row, its 32 lanes
// the 32 values of a block, so the weights and the activations are read coalesced; the activations are staged through shared memory (they
// are shared by the 8 rows of the block); the sum over the row is made once, at the end.
template <int T, int NT>
__global__ void __launch_bounds__(kMmThreads)
k_mm(const uint8_t* __restrict__ w, long long row_bytes, int n_in, int n_out, const float* __restrict__ x, int ldx, int nt_total,
     float* __restrict__ y, int ldy, int y_off) {
    __shared__ float xs[NT][kMmTile];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t0 = (int) blockIdx.y * NT;
    const int nt = min(NT, nt_total - t0);
    const int row = (int) blockIdx.x * kMmWarps + warp;
    const bool valid = row < n_out;
    const uint8_t* rp = w + (size_t) (valid ? row : 0) * (size_t) row_bytes;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int k0 = 0; k0 < n_in; k0 += kMmTile) {
        const int cols = min(kMmTile, n_in - k0);
        for (int idx = tid; idx < NT * kMmTile; idx += kMmThreads) {
            const int t = idx / kMmTile, c = idx - t * kMmTile;
            xs[t][c] = (t < nt && c < cols) ? x[(size_t) (t0 + t) * (size_t) ldx + (size_t) (k0 + c)] : 0.f;
        }
        __syncthreads();
        if (valid) {
            const long long u0 = k0 / 32;
            for (int uu = 0; uu < cols / 32; ++uu) {
                const float wv = g4::dequant_elem<T>(rp, u0 + uu, lane);
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[t] += wv * xs[t][uu * 32 + lane];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = warp_sum(acc[t]);
    if (lane == 0 && valid) {
        for (int t = 0; t < nt; ++t) y[(size_t) (t0 + t) * (size_t) ldy + (size_t) (y_off + row)] = acc[t];
    }
}

/// The same for the experts: grid.y is a group of up to NT (token, expert) pairs that share an expert; group g multiplies the rows of expert
/// grp_expert[g] by the activations rows ent_in[grp_first[g] + i] and writes row ent_out[grp_first[g] + i] of y.
template <int T, int NT>
__global__ void __launch_bounds__(kMmThreads)
k_mm_exp(const uint8_t* __restrict__ w, long long expert_bytes, long long row_bytes, int n_in, int n_out, const float* __restrict__ x, int ldx,
         const int* __restrict__ grp_expert, const int* __restrict__ grp_first, const int* __restrict__ grp_n, const int* __restrict__ ent_in,
         const int* __restrict__ ent_out, float* __restrict__ y, int ldy, int y_off) {
    __shared__ float xs[NT][kMmTile];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = (int) blockIdx.y;
    const int e = grp_expert[g], first = grp_first[g], nt = grp_n[g];
    const int row = (int) blockIdx.x * kMmWarps + warp;
    const bool valid = row < n_out;
    const uint8_t* rp = w + (size_t) e * (size_t) expert_bytes + (size_t) (valid ? row : 0) * (size_t) row_bytes;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int k0 = 0; k0 < n_in; k0 += kMmTile) {
        const int cols = min(kMmTile, n_in - k0);
        for (int idx = tid; idx < NT * kMmTile; idx += kMmThreads) {
            const int t = idx / kMmTile, c = idx - t * kMmTile;
            xs[t][c] = (t < nt && c < cols) ? x[(size_t) ent_in[first + t] * (size_t) ldx + (size_t) (k0 + c)] : 0.f;
        }
        __syncthreads();
        if (valid) {
            const long long u0 = k0 / 32;
            for (int uu = 0; uu < cols / 32; ++uu) {
                const float wv = g4::dequant_elem<T>(rp, u0 + uu, lane);
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[t] += wv * xs[t][uu * 32 + lane];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = warp_sum(acc[t]);
    if (lane == 0 && valid) {
        for (int t = 0; t < nt; ++t) y[(size_t) ent_out[first + t] * (size_t) ldy + (size_t) (y_off + row)] = acc[t];
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w   (w may be null), one block per row; y may be x
__global__ void __launch_bounds__(kThreads)
k_rms(const float* x, int ldx, const float* w, float* y, int ldy, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * (size_t) ldx;
    float ss = 0.f;
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) ss += xr[i] * xr[i];
    const float tot = block_sum(ss, sh);
    const float scale = 1.0f / sqrtf(tot / (float) n + eps);
    float* yr = y + (size_t) blockIdx.x * (size_t) ldy;
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
}

/// out[r] = res[r] + rms(x[r]) * w : the attention's output norm and its residual.  out may be res or x.
__global__ void __launch_bounds__(kThreads)
k_rms_add(const float* x, const float* w, const float* res, float* out, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * (size_t) n;
    float ss = 0.f;
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) ss += xr[i] * xr[i];
    const float tot = block_sum(ss, sh);
    const float scale = 1.0f / sqrtf(tot / (float) n + eps);
    const float* rr = res + (size_t) blockIdx.x * (size_t) n;
    float* o = out + (size_t) blockIdx.x * (size_t) n;
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) o[i] = rr[i] + xr[i] * scale * w[i];
}

/// The end of a layer: cur = rms(mlp) * w1 + rms(moe) * w2  (or just mlp when the layer has no experts: moe null); h = (rms(cur) * w3 + h1) * out_scale
__global__ void __launch_bounds__(kThreads)
k_ffn_combine(const float* mlp, const float* moe, const float* h1, const float* w1, const float* w2, const float* w3, float out_scale,
              float* h, int n, float eps) {
    __shared__ float sh[32];
    __shared__ float cur[8192];
    const size_t o = (size_t) blockIdx.x * (size_t) n;
    float s1 = 0.f, s2 = 0.f;
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) {
        s1 += mlp[o + i] * mlp[o + i];
        if (moe) s2 += moe[o + i] * moe[o + i];
    }
    const float t1 = block_sum(s1, sh);
    const float t2 = moe ? block_sum(s2, sh) : 0.f;
    const float sc1 = 1.0f / sqrtf(t1 / (float) n + eps);
    const float sc2 = moe ? 1.0f / sqrtf(t2 / (float) n + eps) : 0.f;
    float s3 = 0.f;
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) {
        const float c = moe ? (mlp[o + i] * sc1 * w1[i] + moe[o + i] * sc2 * w2[i]) : mlp[o + i];
        cur[i] = c;
        s3 += c * c;
    }
    const float t3 = block_sum(s3, sh);
    const float sc3 = 1.0f / sqrtf(t3 / (float) n + eps);
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) h[o + i] = (cur[i] * sc3 * w3[i] + h1[o + i]) * out_scale;
}

/// out[r][i] = gelu(gate[r][i]) * up[r][i]
__global__ void __launch_bounds__(kThreads)
k_geglu(const float* gate, int ldg, const float* up, int ldu, float* out, int ldo, int n) {
    const float* g = gate + (size_t) blockIdx.y * (size_t) ldg;
    const float* u = up + (size_t) blockIdx.y * (size_t) ldu;
    float* o = out + (size_t) blockIdx.y * (size_t) ldo;
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) o[i] = gelu(g[i]) * u[i];
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// The router of a mixture-of-experts layer, a block per token: r = rms(h1) / sqrt(n_embd) * router_scale; logits = router . r; softmax; the
/// n_used best (the lowest index among equals) with their weights divided by their sum (at least 6.1e-5).
__global__ void __launch_bounds__(kThreads)
k_router(const float* h1, const float* router_scale, const float* router, int n_embd, int n_expert, int n_used, float eps, int* topk_idx,
         float* topk_w) {
    __shared__ float r[8192];
    __shared__ float sh[32];
    __shared__ float lg[kMaxExperts];
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31, nw = (int) blockDim.x >> 5;
    const float* hr = h1 + (size_t) blockIdx.x * (size_t) n_embd;
    float ss = 0.f;
    for (int i = tid; i < n_embd; i += (int) blockDim.x) ss += hr[i] * hr[i];
    const float tot = block_sum(ss, sh);
    const float scale = 1.0f / sqrtf(tot / (float) n_embd + eps);
    const float inv_sqrt = 1.0f / sqrtf((float) n_embd);
    for (int i = tid; i < n_embd; i += (int) blockDim.x) r[i] = hr[i] * scale * inv_sqrt * router_scale[i];
    __syncthreads();
    for (int e = warp; e < n_expert; e += nw) {
        float a = 0.f;
        for (int i = lane; i < n_embd; i += 32) a += router[(size_t) e * (size_t) n_embd + (size_t) i] * r[i];
        a = warp_sum(a);
        if (lane == 0) lg[e] = a;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = lg[0];
        for (int e = 1; e < n_expert; ++e) mx = fmaxf(mx, lg[e]);
        float sum = 0.f;
        for (int e = 0; e < n_expert; ++e) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
        for (int e = 0; e < n_expert; ++e) lg[e] = lg[e] / sum;
        float wsum = 0.f;
        int* oi = topk_idx + (size_t) blockIdx.x * (size_t) n_used;
        float* ow = topk_w + (size_t) blockIdx.x * (size_t) n_used;
        for (int k = 0; k < n_used; ++k) {
            int best = 0;
            float bv = -1.f;
            for (int e = 0; e < n_expert; ++e)
                if (lg[e] > bv) { bv = lg[e]; best = e; }
            oi[k] = best;
            ow[k] = bv;
            wsum += bv;
            lg[best] = -2.f;                  // taken
        }
        wsum = fmaxf(wsum, 6.103515625e-5f);
        for (int k = 0; k < n_used; ++k) ow[k] = ow[k] / wsum;
    }
}

/// moe[t][i] = sum over the token's experts k of d[t * n_used + k][i] * down_scale[expert] * weight  (down_scale may be null)
__global__ void __launch_bounds__(kThreads)
k_expert_sum(const float* d, const int* topk_idx, const float* topk_w, const float* down_scale, int n_used, int n, float* moe) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const int t = (int) blockIdx.y;
    float acc = 0.f;
    for (int k = 0; k < n_used; ++k) {
        const float sc = down_scale ? down_scale[topk_idx[(size_t) t * (size_t) n_used + (size_t) k]] : 1.f;
        acc += d[((size_t) t * (size_t) n_used + (size_t) k) * (size_t) n + (size_t) i] * sc * topk_w[(size_t) t * (size_t) n_used + (size_t) k];
    }
    moe[(size_t) t * (size_t) n + (size_t) i] = acc;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// The head norms, the rotary position and the cache write of the T projected tokens, a block per (token, head): grid (T, n_head + n_kv).
/// A query head: rms * q_norm, rope, written back over its projection.  A KV head: its K (rms * k_norm, rope) and its V (rms, no weight, no
/// rope; from `vraw`, which is `kraw` itself on a layer without attn_v) go to the cache as halves at cell `pos % ring`.
__global__ void __launch_bounds__(kThreads)
k_qkv_post(float* q, const float* kraw, const float* vraw, const float* q_norm, const float* k_norm, const float* ff, uint16_t* kc, uint16_t* vc,
           int hd, int n_head, int n_kv, int n_rot, float base, float eps, int pos0, int ring, int q_dim, int kv_dim) {
    __shared__ float vec[kMaxHead];
    __shared__ float sh[32];
    const int tid = (int) threadIdx.x, t = (int) blockIdx.x, hh = (int) blockIdx.y;
    const int pos = pos0 + t;
    const bool is_q = hh < n_head;
    const int kk = hh - n_head;
    for (int pass = 0; pass < (is_q ? 1 : 2); ++pass) {
        const float* src = is_q ? q + (size_t) t * (size_t) q_dim + (size_t) hh * (size_t) hd
                                : (pass == 0 ? kraw : vraw) + (size_t) t * (size_t) kv_dim + (size_t) kk * (size_t) hd;
        const float* nw = is_q ? q_norm : (pass == 0 ? k_norm : nullptr);
        float ss = 0.f;
        for (int i = tid; i < hd; i += (int) blockDim.x) ss += src[i] * src[i];
        const float tot = block_sum(ss, sh);
        const float scale = 1.0f / sqrtf(tot / (float) hd + eps);
        for (int i = tid; i < hd; i += (int) blockDim.x) vec[i] = nw ? src[i] * scale * nw[i] : src[i] * scale;
        __syncthreads();
        const bool rotate = is_q || pass == 0;
        const int half = n_rot / 2;
        if (is_q) {
            float* dst = q + (size_t) t * (size_t) q_dim + (size_t) hh * (size_t) hd;
            for (int i = tid; i < hd; i += (int) blockDim.x) {
                if (rotate && i < n_rot) {
                    const int p = i < half ? i : i - half;
                    float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
                    if (ff) theta /= ff[p];
                    const float c = cosf(theta), s = sinf(theta);
                    dst[i] = i < half ? vec[i] * c - vec[i + half] * s : vec[i - half] * s + vec[i] * c;
                } else {
                    dst[i] = vec[i];
                }
            }
        } else {
            uint16_t* dst = (pass == 0 ? kc : vc) + ((size_t) kk * (size_t) ring + (size_t) (pos % ring)) * (size_t) hd;
            for (int i = tid; i < hd; i += (int) blockDim.x) {
                float o = vec[i];
                if (rotate && i < n_rot) {
                    const int p = i < half ? i : i - half;
                    float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
                    if (ff) theta /= ff[p];
                    const float c = cosf(theta), s = sinf(theta);
                    o = i < half ? vec[i] * c - vec[i + half] * s : vec[i - half] * s + vec[i] * c;
                }
                dst[i] = g4::float_to_half(o);
            }
        }
        __syncthreads();
    }
}

/// Attention of the T tokens at positions pos0 .. pos0 + T - 1, a block per (token, query head): grid (T, n_head), 4 warps.  Each warp takes
/// every fourth key of the token's window (the last `window` cells for a sliding-window layer, every cell for a global one), keeps an online
/// softmax (max, sum, and the weighted sum of V spread over its lanes: lane l holds elements 2 (l + 32 s) and +1) and the four are merged at the
/// end.  The scale is 1 (the head norms carry it).  K / V rows are read as pairs of halves: a warp reads 128 contiguous bytes at a time.
/// NP = head_dim / 64.
template <int NP>
__global__ void __launch_bounds__(kThreads)
k_attn(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, int n_head, int n_kv, int pos0, int ring, int window, int q_dim) {
    __shared__ float wm[4], wl[4];
    __shared__ float wacc[4][NP * 64];
    constexpr int HD = NP * 64;
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t = (int) blockIdx.x, hh = (int) blockIdx.y;
    const int kk = hh / (n_head / n_kv);
    const int p = pos0 + t;
    const int lo = window > 0 ? max(0, p - window + 1) : 0;
    const float* qh = q + (size_t) t * (size_t) q_dim + (size_t) hh * (size_t) HD;
    float qv[NP * 2];
#pragma unroll
    for (int s = 0; s < NP; ++s) {
        qv[2 * s] = qh[2 * (lane + 32 * s)];
        qv[2 * s + 1] = qh[2 * (lane + 32 * s) + 1];
    }
    float acc[NP * 2];
#pragma unroll
    for (int i = 0; i < NP * 2; ++i) acc[i] = 0.f;
    float m = -INFINITY, l = 0.f;
    const uint16_t* kbase = kc + (size_t) kk * (size_t) ring * (size_t) HD;
    const uint16_t* vbase = vc + (size_t) kk * (size_t) ring * (size_t) HD;
    for (int j = lo + warp; j <= p; j += 4) {
        const size_t cell = (size_t) (j % ring);
        const uint32_t* kr = reinterpret_cast<const uint32_t*>(kbase + cell * (size_t) HD);
        float sc = 0.f;
#pragma unroll
        for (int s = 0; s < NP; ++s) {
            const uint32_t pk = kr[lane + 32 * s];
            sc += qv[2 * s] * g4::half_to_float((uint16_t) (pk & 0xffffu)) + qv[2 * s + 1] * g4::half_to_float((uint16_t) (pk >> 16));
        }
        sc = warp_sum(sc);
        const float mn = fmaxf(m, sc);
        const float a = m == -INFINITY ? 0.f : expf(m - mn);
        const float b = expf(sc - mn);
        l = l * a + b;
        const uint32_t* vr = reinterpret_cast<const uint32_t*>(vbase + cell * (size_t) HD);
#pragma unroll
        for (int s = 0; s < NP; ++s) {
            const uint32_t pv = vr[lane + 32 * s];
            acc[2 * s] = acc[2 * s] * a + b * g4::half_to_float((uint16_t) (pv & 0xffffu));
            acc[2 * s + 1] = acc[2 * s + 1] * a + b * g4::half_to_float((uint16_t) (pv >> 16));
        }
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
#pragma unroll
    for (int s = 0; s < NP; ++s) {
        wacc[warp][2 * (lane + 32 * s)] = acc[2 * s];
        wacc[warp][2 * (lane + 32 * s) + 1] = acc[2 * s + 1];
    }
    __syncthreads();
    float M = wm[0];
    for (int w = 1; w < 4; ++w) M = fmaxf(M, wm[w]);
    float L = 0.f;
    for (int w = 0; w < 4; ++w) L += wm[w] == -INFINITY ? 0.f : wl[w] * expf(wm[w] - M);
    const float inv = 1.0f / L;
    float* o = out + (size_t) t * (size_t) q_dim + (size_t) hh * (size_t) HD;
    for (int e = tid; e < HD; e += (int) blockDim.x) {
        float a = 0.f;
        for (int w = 0; w < 4; ++w) a += wm[w] == -INFINITY ? 0.f : wacc[w][e] * expf(wm[w] - M);
        o[e] = a * inv;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
/// x[i] = cap * tanh(x[i] / cap)
__global__ void __launch_bounds__(kThreads) k_softcap(float* x, int n, float cap) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

/// The largest value and its (lowest) index of x, in two passes: k_argmax_part (a partial of 1024 values per block), k_argmax_final (one block)
__global__ void __launch_bounds__(kThreads) k_argmax_part(const float* x, int n, float* pv, int* pi) {
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    const int tid = (int) threadIdx.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = (int) blockIdx.x * 1024 + tid; i < min(n, ((int) blockIdx.x + 1) * 1024); i += (int) blockDim.x)
        if (x[i] > bv || (x[i] == bv && i < bi)) { bv = x[i]; bi = i; }
    sv[tid] = bv;
    si[tid] = bi;
    __syncthreads();
    for (int s = (int) blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid]))) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { pv[blockIdx.x] = sv[0]; pi[blockIdx.x] = si[0]; }
}
__global__ void __launch_bounds__(kThreads) k_argmax_final(const float* pv, const int* pi, int np, int* out) {
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    const int tid = (int) threadIdx.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = tid; i < np; i += (int) blockDim.x)
        if (pv[i] > bv || (pv[i] == bv && pi[i] < bi)) { bv = pv[i]; bi = pi[i]; }
    sv[tid] = bv;
    si[tid] = bi;
    __syncthreads();
    for (int s = (int) blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid]))) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *out = si[0];
}

}  // namespace g4k
