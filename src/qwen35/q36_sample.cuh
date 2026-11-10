// q36_sample.cuh - the sampler's device half for the Qwen3.6 engine (q36_engine.cu): the repeat / frequency / presence
// penalties applied to the logit rows on the card from a (token, count) list, then the top k candidates of each row
// (k <= 64, best first, equal values ordered by token id).  The host copies back k pairs per row instead of the 248,320
// floats of the row and draws among them (sampling.hpp: sample_from_topk).  The arithmetic of the penalties is
// apply_penalties' operation for operation (no FMA, round-to-nearest).  tests/core/q36_sample_test.cpp runs these kernels
// on the CPU emulation of CUDA (tests/core/cuda_cpu_emu.hpp) against the host reference.
#pragma once

#include "strata/qwen35/sampling.hpp"

#include <cfloat>
#include <cstdint>

#if defined(STRATA_Q36_EMU)
#define Q36S_DIV(a, b) ((a) / (b))
#define Q36S_MUL(a, b) ((a) * (b))
#define Q36S_ADD(a, b) ((a) + (b))
#define Q36S_SUB(a, b) ((a) - (b))
#define Q36S_LAUNCH(kernel, grid, block, stream, ...) g4emu::launch(dim3(grid), dim3(block), [&] { kernel(__VA_ARGS__); })
#else
#include <cuda_runtime.h>
#define Q36S_DIV(a, b) __fdiv_rn((a), (b))
#define Q36S_MUL(a, b) __fmul_rn((a), (b))
#define Q36S_ADD(a, b) __fadd_rn((a), (b))
#define Q36S_SUB(a, b) __fsub_rn((a), (b))
#define Q36S_LAUNCH(kernel, grid, block, stream, ...) kernel<<<dim3(grid), dim3(block), 0, stream>>>(__VA_ARGS__)
#endif

namespace q36s {

using strata::qwen35::PenEntry;

constexpr int kMaxK = 64;           // candidates per row
constexpr int kPartThreads = 256;   // stage 1: a block of 256 threads, 8 elements each
constexpr int kPartElems = kPartThreads * 8;
constexpr int kFinalThreads = 1024;
constexpr int kMaxDelta = 64;       // delta entries per row (a verify window adds at most 2 x 8 tokens' worth)

__device__ __forceinline__ float penalise(float v, int cnt, float repeat, float freq, float present) {
    if (repeat != 1.0f) v = v > 0.0f ? Q36S_DIV(v, repeat) : Q36S_MUL(v, repeat);
    return Q36S_SUB(v, Q36S_ADD(Q36S_MUL((float) cnt, freq), present));
}

/// The base list on every row: entry e of `base` on row r, unless row r's delta list names its id.  `rows` rows of `ld`
/// floats, `n` valid; the deltas of row r are deltas[row_off[r] .. row_off[r + 1]).  grid (ceil(nb / 256), rows), block 256.
__global__ void k_penalize_base(float* lg, int ld, int n, const PenEntry* base, int nb, const PenEntry* deltas,
                                const int32_t* row_off, float repeat, float freq, float present) {
    __shared__ int32_t dids[kMaxDelta];
    const int r = (int) blockIdx.y;
    const int d0 = row_off[r], nd = row_off[r + 1] - d0;
    if ((int) threadIdx.x < nd) dids[threadIdx.x] = deltas[d0 + threadIdx.x].id;
    __syncthreads();
    const int e = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= nb) return;
    const PenEntry p = base[e];
    if (p.id < 0 || p.id >= n || p.cnt <= 0) return;
    for (int i = 0; i < nd; ++i)
        if (dids[i] == p.id) return;
    float* v = lg + (size_t) r * ld + p.id;
    *v = penalise(*v, p.cnt, repeat, freq, present);
}

/// The delta list of each row (final counts; 0: nothing).  grid (rows), block 64.
__global__ void k_penalize_delta(float* lg, int ld, int n, const PenEntry* deltas, const int32_t* row_off, float repeat,
                                 float freq, float present) {
    const int r = (int) blockIdx.x;
    const int d0 = row_off[r], nd = row_off[r + 1] - d0;
    if ((int) threadIdx.x >= nd) return;
    const PenEntry p = deltas[d0 + threadIdx.x];
    if (p.id < 0 || p.id >= n || p.cnt <= 0) return;
    float* v = lg + (size_t) r * ld + p.id;
    *v = penalise(*v, p.cnt, repeat, freq, present);
}

struct Cand {
    float v;
    int32_t id;   // -1: none
};
__device__ __forceinline__ bool better(const Cand& a, const Cand& b) {
    if (b.id < 0) return a.id >= 0;
    if (a.id < 0) return false;
    return a.v > b.v || (a.v == b.v && a.id < b.id);
}
__device__ __forceinline__ Cand warp_best(Cand c) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        Cand d;
        d.v = __shfl_xor_sync(0xffffffffu, c.v, o);
        d.id = __shfl_xor_sync(0xffffffffu, c.id, o);
        if (better(d, c)) c = d;
    }
    return c;
}

/// Stage 1: block (b, r) takes the elements [b * 2048, b * 2048 + 2048) of row r and writes its k best, best first, at
/// cv / ci [r][b * k + i] (id -1 past the elements it has).  grid (ceil(n / 2048), rows), block 256.
__global__ void __launch_bounds__(kPartThreads) k_topk_part(const float* lg, int ld, int n, int k, float* cv, int32_t* ci) {
    __shared__ Cand wb[kPartThreads / 32];
    __shared__ Cand pick;
    const int r = (int) blockIdx.y, b = (int) blockIdx.x;
    const float* row = lg + (size_t) r * ld;
    Cand mine[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = b * kPartElems + j * kPartThreads + (int) threadIdx.x;
        mine[j].id = i < n ? i : -1;
        mine[j].v = i < n ? row[i] : 0.0f;
    }
    const int lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5;
    for (int round = 0; round < k; ++round) {
        Cand best = mine[0];
#pragma unroll
        for (int j = 1; j < 8; ++j)
            if (better(mine[j], best)) best = mine[j];
        best = warp_best(best);
        if (lane == 0) wb[warp] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            Cand c = wb[0];
            for (int w = 1; w < kPartThreads / 32; ++w)
                if (better(wb[w], c)) c = wb[w];
            pick = c;
        }
        __syncthreads();
        const Cand c = pick;
        if (c.id >= 0) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (mine[j].id == c.id) mine[j].id = -1;
        }
        if (threadIdx.x == 0) {
            cv[((size_t) r * gridDim.x + b) * k + round] = c.v;
            ci[((size_t) r * gridDim.x + b) * k + round] = c.id;
        }
        __syncthreads();   // (pick is rewritten next round)
    }
}

/// Stage 2: the k best of row r's `nc` candidates (cv / ci [r][nc]), best first, into ov / oi [r][k].  grid (rows), block 1024;
/// nc <= 8 * 1024.
__global__ void __launch_bounds__(kFinalThreads) k_topk_final(const float* cv, const int32_t* ci, int nc, int k, float* ov, int32_t* oi) {
    __shared__ Cand wb[kFinalThreads / 32];
    __shared__ Cand pick;
    const int r = (int) blockIdx.x;
    Cand mine[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = j * kFinalThreads + (int) threadIdx.x;
        mine[j].id = i < nc ? ci[(size_t) r * nc + i] : -1;
        mine[j].v = i < nc ? cv[(size_t) r * nc + i] : 0.0f;
    }
    const int lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5;
    for (int round = 0; round < k; ++round) {
        Cand best = mine[0];
#pragma unroll
        for (int j = 1; j < 8; ++j)
            if (better(mine[j], best)) best = mine[j];
        best = warp_best(best);
        if (lane == 0) wb[warp] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            Cand c = wb[0];
            for (int w = 1; w < kFinalThreads / 32; ++w)
                if (better(wb[w], c)) c = wb[w];
            pick = c;
        }
        __syncthreads();
        const Cand c = pick;
        if (c.id >= 0) {
            int hit = -1;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (mine[j].id == c.id && hit < 0) hit = j;   // (one copy of an id per thread: stage 1 writes each once)
            if (hit >= 0) mine[hit].id = -1;
        }
        if (threadIdx.x == 0) {
            ov[(size_t) r * k + round] = c.v;
            oi[(size_t) r * k + round] = c.id;
        }
        __syncthreads();
    }
}

/// stage 1 blocks per row, and the candidate count stage 2 reads
inline int part_blocks(int n) { return (n + kPartElems - 1) / kPartElems; }

/// The device buffers of the sampler (the engine allocates them: see q36_engine.cu).
struct Bufs {
    PenEntry* base = nullptr;     // [cap_base]
    PenEntry* deltas = nullptr;   // [rows * kMaxDelta]
    int32_t* row_off = nullptr;   // [rows + 1]
    float* cv = nullptr;          // [rows][part_blocks * k]
    int32_t* ci = nullptr;
    float* ov = nullptr;          // [rows][k]
    int32_t* oi = nullptr;
};

/// The penalties (nb base entries, the rows' deltas) then the top k of `rows` rows of `lg` (ld apart, n valid) into B.ov /
/// B.oi.  The lists are on the device already.  k <= kMaxK; n <= 8 * 1024 * 2048.
template <class Stream>
inline void run(const Bufs& B, float* lg, int ld, int n, int rows, int k, int nb, bool penalise, float repeat, float freq,
                float present, Stream st) {
    if (penalise) {
        if (nb > 0)
            Q36S_LAUNCH(k_penalize_base, dim3((unsigned) ((nb + 255) / 256), (unsigned) rows), dim3(256), st, lg, ld, n, B.base, nb,
                        B.deltas, B.row_off, repeat, freq, present);
        Q36S_LAUNCH(k_penalize_delta, dim3((unsigned) rows), dim3(kMaxDelta), st, lg, ld, n, B.deltas, B.row_off, repeat, freq, present);
    }
    const int pb = part_blocks(n);
    Q36S_LAUNCH(k_topk_part, dim3((unsigned) pb, (unsigned) rows), dim3(kPartThreads), st, lg, ld, n, k, B.cv, B.ci);
    Q36S_LAUNCH(k_topk_final, dim3((unsigned) rows), dim3(kFinalThreads), st, B.cv, B.ci, pb * k, k, B.ov, B.oi);
}

}  // namespace q36s
