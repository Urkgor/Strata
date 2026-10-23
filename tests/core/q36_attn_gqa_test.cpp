// q36_attn.cuh's k_attn_part_gqa against a copy of q36_engine.cu's k_attn_part, on the CPU: every CUDA thread is a std::thread, a
// block is 256 of them, __syncthreads is a barrier and __shared__ is static (blocks run one after the other).  The outputs of the
// two kernels (pacc, pm, pl for every query row and chunk) must be equal bit for bit, for decode rows, for verify windows
// (several tokens), for sessions picked per row (RowSel), and for a context that is not a multiple of the chunk, so a cell past
// the cache slab would be read by an unguarded load.  Built with -ffp-contract=off so a fused multiply-add cannot differ between
// the two.  No GPU.  (It checks the indexing, the barriers and the arithmetic order of the kernel; speed is a GPU matter.)
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

// ---- a CUDA block on the CPU
struct Dim3 { int x = 0, y = 0, z = 0; };
thread_local Dim3 threadIdx, blockIdx, blockDim;
struct float4 { float x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
std::barrier<>* g_barrier = nullptr;
#define __global__
#define __device__
#define __shared__ static
#define __launch_bounds__(n)
#define __syncthreads() g_barrier->arrive_and_wait()

template <class F, class... A> void launch(int gx, int gy, int threads, F kernel, A... args) {
    std::barrier<> bar(threads);
    g_barrier = &bar;
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx) {
            std::vector<std::thread> ts;
            for (int tx = 0; tx < threads; ++tx)
                ts.emplace_back([=] {
                    threadIdx = {tx, 0, 0};
                    blockIdx = {bx, by, 0};
                    blockDim = {threads, 1, 1};
                    kernel(args...);
                });
            for (auto& t : ts) t.join();
        }
    g_barrier = nullptr;
}

#include "q36_attn.cuh"

// ---- the old kernel, copied from q36_engine.cu (k_attn_part, block_sum, block_max, RowSel)
constexpr int kThreads = 256, kChunkCells = 256;
struct RowSel { const int32_t* cell = nullptr; const int32_t* slot = nullptr; const int32_t* delta = nullptr;
                float* const* kc = nullptr; float* const* vc = nullptr; };
float block_sum(float v, float* sh) {
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
float block_max(float v, float* sh) {
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
void k_attn_part(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                 int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const float* kp = kc + ((size_t) kvh * ctx + p) * HD;
        float a = 0.f;
        for (int i = 0; i < HD; ++i) a += qs[i] * kp[i];
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = std::min(kChunkCells, npos - p0);
    float acc = 0.f;
    const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// one comparison: T tokens from t0, `ctx` cells, the first of them at pos0 (or `cells` per row with `sessions` caches)
static bool compare(int T, int pos0v, int ctx, bool per_row, unsigned seed, const char* what) {
    constexpr int H = 16, HKV = 2, HD = 256, G = 8;
    const int n_chunk = (ctx + 255) / 256;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    const int sessions = per_row ? 3 : 1;
    // exact-size allocations: an unguarded read past a KV head's slab leaves the vector (address sanitizer, when it is on)
    std::vector<std::vector<float>> K(sessions), V(sessions);
    for (int s = 0; s < sessions; ++s) {
        K[s].resize((size_t) HKV * ctx * HD);
        V[s].resize((size_t) HKV * ctx * HD);
        for (auto& v : K[s]) v = g(rng) * 0.3f;
        for (auto& v : V[s]) v = g(rng);
    }
    std::vector<float> q((size_t) T * H * HD);
    for (auto& v : q) v = g(rng);
    int32_t pos0 = pos0v;
    std::vector<int32_t> cell(T), slot(T);
    std::vector<float*> kcs(sessions), vcs(sessions);
    for (int s = 0; s < sessions; ++s) { kcs[s] = K[s].data(); vcs[s] = V[s].data(); }
    for (int t = 0; t < T; ++t) { cell[t] = per_row ? (pos0v + 37 * t) % (ctx - 1) : 0; slot[t] = t % sessions; }
    const float scale = 1.0f / std::sqrt((float) HD);
    const size_t rows = (size_t) T * H;
    std::vector<float> acc_a(rows * n_chunk * HD, -7.f), acc_b(rows * n_chunk * HD, -7.f);
    std::vector<float> m_a(rows * n_chunk, -7.f), m_b(rows * n_chunk, -7.f), l_a(rows * n_chunk, -7.f), l_b(rows * n_chunk, -7.f);
    RowSel rs;
    if (per_row) { rs.cell = cell.data(); rs.slot = slot.data(); rs.kc = kcs.data(); rs.vc = vcs.data(); }
    const float* kc = per_row ? nullptr : K[0].data();
    const float* vc = per_row ? nullptr : V[0].data();
    launch(n_chunk, T * H, 256, k_attn_part, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx, scale,
           n_chunk, acc_a.data(), m_a.data(), l_a.data(), rs);
    launch(n_chunk, T * HKV, 256, q36k::k_attn_part_gqa, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx,
           scale, n_chunk, acc_b.data(), m_b.data(), l_b.data(), rs.cell, rs.slot, rs.kc, rs.vc);
    static_assert(G == q36k::kAttnGroup, "H / HKV");
    // chunks past a row's cells are never written by either kernel (the merge stops before them): both keep the fill value
    const bool same = std::memcmp(acc_a.data(), acc_b.data(), acc_a.size() * 4) == 0 &&
                      std::memcmp(m_a.data(), m_b.data(), m_a.size() * 4) == 0 &&
                      std::memcmp(l_a.data(), l_b.data(), l_a.size() * 4) == 0;
    check(same, what);
    return same;
}

int main() {
    std::printf("k_attn_part_gqa against k_attn_part, bit for bit (CPU threads)\n");
    compare(1, 600, 700, false, 1, "decode, 1 token at cell 600, context 700 (2 full chunks + a partial one)");
    compare(1, 255, 512, false, 2, "decode at cell 255: the last cell of chunk 0, context 512");
    compare(1, 256, 512, false, 3, "decode at cell 256: the first cell of chunk 1");
    compare(1, 0, 300, false, 4, "the first token: one cell");
    compare(4, 300, 777, false, 5, "a verify window of 4 tokens");
    compare(3, 50, 900, true, 6, "3 rows on 3 sessions, per-row cells (RowSel)");
    compare(1, 699, 700, false, 7, "the last cell of a 700-cell cache: the K read of the partial chunk is guarded");
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
