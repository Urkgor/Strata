// q36_attn.cuh - the decode attention of the Qwen3.6 engine (q36_engine.cu) for grouped-query attention: one block serves
// the 8 query heads that share a KV head, and reads K through shared memory, coalesced.
//
// WHY.  k_attn_part (q36_engine.cu) gives every (chunk, query head) its own block, and in it thread j walks the K row of
// cell p0 + j with a scalar loop: the 32 lanes of a warp read 32 rows 1 KB apart, one float each, and on a GTX 10 card (sm_61)
// global loads do not go through the L1 by default, so every one of those loads is its own L2 sector request.  A GTX 1080 Ti
// decoded 60 tokens/s at a short context and 6 at 25,000 tokens.  The 8 query heads of a KV head also each re-read the
// same K and V from memory.
//
// WHAT.  The arithmetic of every head is the old kernel's, operation for operation: the score is the same sequential sum over
// the 256 dimensions, the max and the sum of the chunk come from the same block_max / block_sum, the V sum runs over the
// chunk's cells in the same order.  Only the way K reaches the registers changes (a 256-cell x 16-dimension slice at a time,
// loaded as float4 into shared memory with a 17-float row stride, so the 256 threads read it without bank conflicts) and
// that the K / V of a cell are fetched once for the 8 heads.  tests/core/q36_attn_gqa_test.cpp runs this file's kernel and a
// copy of the old one on the CPU (one thread per CUDA thread) and compares every output bit.
//
// Layout contract (the same as k_attn_part): K / V of a layer [kv head][ctx cells][256] elements (floats, or halves with --kv f16); q [T][H][256]; pacc / pm / pl
// indexed by (row = (t - t0) * H + head) * n_chunk + chunk.  Only H / HKV == 8 and head dimension 256, chunks of 256 cells.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#ifndef Q36_ATTN_NO_CUDA_HEADERS   // the CPU test brings its own __half
#include <cuda_fp16.h>
#endif

namespace q36k {

// The K / V cache holds float (KT = float) or half (KT = __half, --kv f16): one element in, one float out, and back.
__device__ __forceinline__ float kv_to_float(float x) { return x; }
__device__ __forceinline__ float kv_to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ void kv_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void kv_store(__half* p, float v) { *p = __float2half(v); }

constexpr int kAttnThreads = 256;   // threads per block == cells per chunk == the head dimension
constexpr int kAttnChunk = 256;
constexpr int kAttnHD = 256;
constexpr int kAttnGroup = 8;       // query heads per KV head this kernel serves
constexpr int kKSlice = 16;         // K dimensions staged in shared memory at a time

// the same two reductions as q36_engine.cu's block_sum / block_max (same order, so the same bits)
__device__ inline float attn_block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ inline float attn_block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// grid (n chunks, rows_t * HKV), block 256.  Block (c, r): token t = t0 + r / HKV, KV head r % HKV, the chunk of cells
/// [c * 256, c * 256 + 256) that the token sees (cells < npos); writes the partials of its 8 heads.  `kc_` / `vc_` (and the
/// session tables) point at arrays of KT: float or __half.
template <class KT>
__global__ void __launch_bounds__(kAttnThreads)
k_attn_part_gqa(const float* q, const float* kc_, const float* vc_, const int32_t* pos0, int t0, int H, int HKV, int ctx,
                float scale, int n_chunk, float* pacc, float* pm, float* pl, const int32_t* rs_cell, const int32_t* rs_slot,
                float* const* rs_kc, float* const* rs_vc) {
    constexpr int HD = kAttnHD, G = kAttnGroup, NC = kAttnChunk, KS = kKSlice;
    __shared__ float qs[G][HD];
    __shared__ float sc[G][NC];
    __shared__ float ks[NC][KS + 1];
    __shared__ float sh[kAttnThreads];
    const int c = blockIdx.x, r = blockIdx.y;
    const int t = t0 + r / HKV, kvh = r % HKV;
    const int npos = (rs_cell ? rs_cell[t] : *pos0 + t) + 1;
    if (rs_kc) {
        kc_ = rs_kc[rs_slot[t]];
        vc_ = rs_vc[rs_slot[t]];
    }
    const KT* kc = reinterpret_cast<const KT*>(kc_);
    const KT* vc = reinterpret_cast<const KT*>(vc_);
    const int p0 = c * NC;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
#pragma unroll
    for (int i = 0; i < G; ++i) qs[i][j] = q[((size_t) t * H + (size_t) kvh * G + i) * HD + j];
    const int p = p0 + j;
    const KT* kbase = kc + ((size_t) kvh * ctx + p0) * HD;
    float a[G];
#pragma unroll
    for (int i = 0; i < G; ++i) a[i] = 0.f;
    for (int d0 = 0; d0 < HD; d0 += KS) {
        __syncthreads();   // the last slice is consumed (the first time: qs is written)
        if constexpr (sizeof(KT) == 4) {
#pragma unroll
            for (int rr = 0; rr < NC * KS / 4 / kAttnThreads; ++rr) {   // the slice: 256 cells x 16 floats, 4 float4 per thread
                const int idx = j + rr * kAttnThreads, cell = idx / (KS / 4), part = idx % (KS / 4);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p0 + cell < npos) v = *reinterpret_cast<const float4*>(kbase + (size_t) cell * HD + d0 + part * 4);
                ks[cell][part * 4 + 0] = v.x;
                ks[cell][part * 4 + 1] = v.y;
                ks[cell][part * 4 + 2] = v.z;
                ks[cell][part * 4 + 3] = v.w;
            }
        } else {
#pragma unroll
            for (int rr = 0; rr < NC * KS / 8 / kAttnThreads; ++rr) {   // 256 cells x 16 halves: 2 x 16 bytes per thread
                const int idx = j + rr * kAttnThreads, cell = idx / (KS / 8), part = idx % (KS / 8);
                uint4 u = make_uint4(0u, 0u, 0u, 0u);
                if (p0 + cell < npos) u = *reinterpret_cast<const uint4*>(kbase + (size_t) cell * HD + d0 + part * 8);
                const KT* h = reinterpret_cast<const KT*>(&u);
#pragma unroll
                for (int k = 0; k < 8; ++k) ks[cell][part * 8 + k] = kv_to_float(h[k]);
            }
        }
        __syncthreads();
        if (p < npos) {
#pragma unroll
            for (int dd = 0; dd < KS; ++dd) {
                const float kv = ks[j][dd];
#pragma unroll
                for (int i = 0; i < G; ++i) a[i] += qs[i][d0 + dd] * kv;
            }
        }
    }
    float m_[G], l_[G];
#pragma unroll
    for (int i = 0; i < G; ++i) {
        const float s = p < npos ? a[i] * scale : -INFINITY;
        const float m = attn_block_max(s, sh);
        const float e = p < npos ? expf(s - m) : 0.f;
        sc[i][j] = e;
        m_[i] = m;
        l_[i] = attn_block_sum(e, sh);   // its syncs also publish sc[i][*] to the V loop below
    }
    const int n = NC < npos - p0 ? NC : npos - p0;
    const KT* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    float acc[G];
#pragma unroll
    for (int i = 0; i < G; ++i) acc[i] = 0.f;
    for (int k = 0; k < n; ++k) {
        const float v = kv_to_float(vp[(size_t) k * HD]);
#pragma unroll
        for (int i = 0; i < G; ++i) acc[i] += sc[i][k] * v;
    }
#pragma unroll
    for (int i = 0; i < G; ++i) {
        const size_t o = ((size_t) (t - t0) * H + (size_t) kvh * G + i) * n_chunk + c;
        pacc[o * HD + j] = acc[i];
        if (j == 0) {
            pm[o] = m_[i];
            pl[o] = l_[i];
        }
    }
}

}  // namespace q36k
