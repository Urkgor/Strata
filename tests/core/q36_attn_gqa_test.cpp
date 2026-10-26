// q36_attn.cuh's k_attn_part_gqa against a copy of q36_engine.cu's k_attn_part, on the CPU: every CUDA thread is a std::thread, a
// block is 256 of them, __syncthreads is a barrier and __shared__ is static (blocks run one after the other).  The outputs of the
// two kernels (pacc, pm, pl for every query row and chunk) must be equal bit for bit, for decode rows, for verify windows
// (several tokens), for sessions picked per row (RowSel), and for a context that is not a multiple of the chunk, so a cell past
// the cache slab would be read by an unguarded load.  Both cache element types: float, and half (--kv f16): there the two kernels
// read the same halves, and the result is also the one the float kernel gives on the half-rounded values (a half cache is exactly
// "the same attention on rounded K / V").  Built with -ffp-contract=off so a fused multiply-add cannot differ between the
// two.  No GPU.  (It checks the indexing, the barriers and the arithmetic order of the kernel; speed is a GPU matter.)
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
struct uint4 { unsigned x, y, z, w; };
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }
// IEEE half, round to nearest even (what __float2half does); q36_attn.cuh finds it through Q36_ATTN_NO_CUDA_HEADERS
struct __half { uint16_t bits; };
inline float __half2float(__half h) {
    const uint32_t s = (uint32_t) (h.bits & 0x8000u) << 16;
    uint32_t e = (h.bits >> 10) & 0x1f, m = h.bits & 0x3ff, u;
    if (e == 0) {
        if (m == 0) u = s;
        else { int sh = 0; while (!(m & 0x400)) { m <<= 1; ++sh; } m &= 0x3ff; u = s | ((113u - sh) << 23) | (m << 13); }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}
inline __half __float2half(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint16_t s = (uint16_t) ((u >> 16) & 0x8000u);
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (((u >> 23) & 0xff) == 0xff) return {(uint16_t) (s | 0x7c00u | (m ? 0x200u : 0u))};
    if (e >= 31) return {(uint16_t) (s | 0x7c00u)};
    if (e <= 0) {
        if (e < -10) return {s};
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return {(uint16_t) (s | r)};
    }
    uint32_t r = ((uint32_t) e << 10) | (m >> 13); const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;   // a carry into the exponent is the right rounding
    return {(uint16_t) (s | r)};
}
#define Q36_ATTN_NO_CUDA_HEADERS
std::barrier<>* g_barrier = nullptr;
#define __global__
#define __device__
#define __forceinline__ inline
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
template <class KT>
void k_attn_part(const float* q, const float* kc_, const float* vc_, const int32_t* pos0, int t0, int H,
                 int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc_ = rs.kc[rs.slot[t]];
        vc_ = rs.vc[rs.slot[t]];
    }
    const KT* kc = reinterpret_cast<const KT*>(kc_);
    const KT* vc = reinterpret_cast<const KT*>(vc_);
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const KT* kp = kc + ((size_t) kvh * ctx + p) * HD;
        float a = 0.f;
        for (int i = 0; i < HD; ++i) a += qs[i] * q36k::kv_to_float(kp[i]);
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = std::min(kChunkCells, npos - p0);
    float acc = 0.f;
    const KT* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * q36k::kv_to_float(vp[(size_t) k * HD]);
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

// one comparison: T tokens from pos0, `ctx` cells, the first of them at pos0 (or `cells` per row with `sessions` caches).
// half_cache: K / V are stored as halves (--kv f16).  Returns whether every kernel pair agreed bit for bit.
static bool compare(int T, int pos0v, int ctx, bool per_row, bool half_cache, unsigned seed, const char* what) {
    constexpr int H = 16, HKV = 2, HD = 256, G = 8;
    const int n_chunk = (ctx + 255) / 256;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    const int sessions = per_row ? 3 : 1;
    // exact-size allocations: an unguarded read past a KV head's slab leaves the vector (address sanitizer, when it is on)
    const size_t n_kv = (size_t) HKV * ctx * HD;
    std::vector<std::vector<float>> K(sessions), V(sessions), Kr(sessions), Vr(sessions);      // floats; Kr / Vr: rounded to half
    std::vector<std::vector<__half>> Kh(sessions), Vh(sessions);
    for (int s = 0; s < sessions; ++s) {
        K[s].resize(n_kv); V[s].resize(n_kv);
        for (auto& v : K[s]) v = g(rng) * 0.3f;
        for (auto& v : V[s]) v = g(rng);
        if (half_cache) {
            Kh[s].resize(n_kv); Vh[s].resize(n_kv); Kr[s].resize(n_kv); Vr[s].resize(n_kv);
            for (size_t i = 0; i < n_kv; ++i) {
                Kh[s][i] = __float2half(K[s][i]); Vh[s][i] = __float2half(V[s][i]);
                Kr[s][i] = __half2float(Kh[s][i]); Vr[s][i] = __half2float(Vh[s][i]);
            }
        }
    }
    std::vector<float> q((size_t) T * H * HD);
    for (auto& v : q) v = g(rng);
    int32_t pos0 = pos0v;
    std::vector<int32_t> cell(T), slot(T);
    for (int t = 0; t < T; ++t) { cell[t] = per_row ? (pos0v + 37 * t) % (ctx - 1) : 0; slot[t] = t % sessions; }
    const float scale = 1.0f / std::sqrt((float) HD);
    const size_t rows = (size_t) T * H;
    // the cache the kernels see, as float* handles (the engine's convention); the half cache's handles point at halves
    auto handles = [&](bool k, bool use_half, bool use_rounded) {
        std::vector<float*> h(sessions);
        for (int s = 0; s < sessions; ++s)
            h[s] = use_half ? reinterpret_cast<float*>((k ? Kh : Vh)[s].data())
                            : (use_rounded ? (k ? Kr : Vr)[s].data() : (k ? K : V)[s].data());
        return h;
    };
    struct Out { std::vector<float> acc, m, l; Out(size_t rows, int nc) : acc(rows * nc * HD, -7.f), m(rows * nc, -7.f), l(rows * nc, -7.f) {} };
    auto run = [&](bool gqa, bool use_half, bool use_rounded) {
        Out o(rows, n_chunk);
        auto kcs = handles(true, use_half, use_rounded), vcs = handles(false, use_half, use_rounded);
        RowSel rs;
        if (per_row) { rs.cell = cell.data(); rs.slot = slot.data(); rs.kc = kcs.data(); rs.vc = vcs.data(); }
        const float* kc = per_row ? nullptr : kcs[0];
        const float* vc = per_row ? nullptr : vcs[0];
        if (!gqa) {
            if (use_half) launch(n_chunk, T * H, 256, k_attn_part<__half>, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx, scale, n_chunk, o.acc.data(), o.m.data(), o.l.data(), rs);
            else launch(n_chunk, T * H, 256, k_attn_part<float>, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx, scale, n_chunk, o.acc.data(), o.m.data(), o.l.data(), rs);
        } else {
            if (use_half) launch(n_chunk, T * HKV, 256, q36k::k_attn_part_gqa<__half>, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx, scale, n_chunk, o.acc.data(), o.m.data(), o.l.data(), rs.cell, rs.slot, rs.kc, rs.vc);
            else launch(n_chunk, T * HKV, 256, q36k::k_attn_part_gqa<float>, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx, scale, n_chunk, o.acc.data(), o.m.data(), o.l.data(), rs.cell, rs.slot, rs.kc, rs.vc);
        }
        return o;
    };
    static_assert(G == q36k::kAttnGroup, "H / HKV");
    // chunks past a row's cells are never written by either kernel (the merge stops before them): both keep the fill value
    auto same = [](const Out& a, const Out& b) {
        return std::memcmp(a.acc.data(), b.acc.data(), a.acc.size() * 4) == 0 && std::memcmp(a.m.data(), b.m.data(), a.m.size() * 4) == 0 &&
               std::memcmp(a.l.data(), b.l.data(), a.l.size() * 4) == 0;
    };
    bool ok;
    if (!half_cache) {
        ok = same(run(false, false, false), run(true, false, false));
    } else {
        const Out old_h = run(false, true, false), new_h = run(true, true, false), old_f_on_rounded = run(false, false, true);
        ok = same(old_h, new_h) && same(old_h, old_f_on_rounded);   // the half kernels agree, and equal float attention on rounded K / V
    }
    check(ok, what);
    return ok;
}


// ---- --kv q8_0: the Q8_0 store (q8_store) against a host reference, and the Q8 kernel against the old float kernel on the
// dequantized values (q * d, d a half): the arithmetic after the dequantization is the same, so every output bit must be.
struct Q8Ref {   // one block of 32: the reference of ggml's quantize_row_q8_0
    int8_t q[32];
    __half d;
};
static Q8Ref q8_ref(const float* x) {
    float amax = 0.f;
    for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(x[k]));
    const float d = amax / 127.0f, id = d != 0.f ? 1.0f / d : 0.f;
    Q8Ref r;
    for (int k = 0; k < 32; ++k) r.q[k] = (int8_t) std::round(x[k] * id);
    r.d = __float2half(d);
    return r;
}
void k_test_q8_store(const float* x, float* qb, int8_t* qs, __half* sc) { q36k::q8_store(x[threadIdx.x], qb, qs, sc); }

static void test_q8_store() {
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.f, 1.f);
    constexpr int HD = 256;
    std::vector<float> x(HD), qb(HD);
    bool ok = true;
    for (int round = 0; round < 12; ++round) {
        for (int i = 0; i < HD; ++i) x[i] = g(rng) * (round % 3 == 0 ? 0.01f : round % 3 == 1 ? 1.f : 300.f);
        if (round == 3) for (int i = 0; i < 32; ++i) x[i] = 0.f;                 // a block of zeros: d = 0, no division
        if (round == 4) for (int i = 32; i < 64; ++i) x[i] = (i % 2 ? 1.f : -1.f) * 127.f;   // |x| = 127 d: exact quants
        if (round == 5) for (int i = 64; i < 96; ++i) x[i] = (float) (i - 63) * 0.5f;       // ties at .5 of a quant step
        std::vector<int8_t> qs(HD, 55);
        std::vector<__half> sc(HD / 32);
        launch(1, 1, HD, k_test_q8_store, (const float*) x.data(), qb.data(), qs.data(), sc.data());
        for (int b = 0; b < HD / 32; ++b) {
            const Q8Ref r = q8_ref(&x[b * 32]);
            if (sc[b].bits != r.d.bits) ok = false;
            for (int k = 0; k < 32; ++k) if (qs[b * 32 + k] != r.q[k]) ok = false;
        }
    }
    check(ok, "q8_store: quants and scales equal the Q8_0 reference (zeros, ties, small, large)");
}

static bool compare_q8(int T, int pos0v, int ctx, bool per_row, unsigned seed, const char* what) {
    constexpr int H = 16, HKV = 2, HD = 256;
    const int n_chunk = (ctx + 255) / 256;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    const int sessions = per_row ? 3 : 1;
    const size_t n_kv = (size_t) HKV * ctx * HD, n_sc = (size_t) HKV * ctx * (HD / 32);
    // per session: the K / V as Q8_0 arrays (quants then scales, one allocation) and as the floats they stand for
    std::vector<std::vector<uint8_t>> Kq(sessions), Vq(sessions);
    std::vector<std::vector<float>> Kd(sessions), Vd(sessions);
    auto build = [&](std::vector<uint8_t>& q, std::vector<float>& d, float amp) {
        std::vector<float> x(n_kv);
        for (auto& v : x) v = g(rng) * amp;
        q.assign(n_kv + n_sc * 2, 0);
        d.assign(n_kv, 0.f);
        __half* sc = reinterpret_cast<__half*>(q.data() + n_kv);
        for (size_t b = 0; b < n_sc; ++b) {
            const Q8Ref r = q8_ref(&x[b * 32]);
            sc[b] = r.d;
            for (int k = 0; k < 32; ++k) {
                q[b * 32 + k] = (uint8_t) r.q[k];
                d[b * 32 + k] = (float) r.q[k] * __half2float(r.d);
            }
        }
    };
    for (int s = 0; s < sessions; ++s) { build(Kq[s], Kd[s], 0.3f); build(Vq[s], Vd[s], 1.f); }
    std::vector<float> q((size_t) T * H * HD);
    for (auto& v : q) v = g(rng);
    int32_t pos0 = pos0v;
    std::vector<int32_t> cell(T), slot(T);
    for (int t = 0; t < T; ++t) { cell[t] = per_row ? (pos0v + 37 * t) % (ctx - 1) : 0; slot[t] = t % sessions; }
    const float scale = 1.0f / std::sqrt((float) HD);
    const size_t rows = (size_t) T * H;
    struct Out { std::vector<float> acc, m, l; Out(size_t rows, int nc) : acc(rows * nc * HD, -7.f), m(rows * nc, -7.f), l(rows * nc, -7.f) {} };
    auto run = [&](bool q8) {
        Out o(rows, n_chunk);
        std::vector<float*> kcs(sessions), vcs(sessions);
        for (int s = 0; s < sessions; ++s) {
            kcs[s] = q8 ? reinterpret_cast<float*>(Kq[s].data()) : Kd[s].data();
            vcs[s] = q8 ? reinterpret_cast<float*>(Vq[s].data()) : Vd[s].data();
        }
        RowSel rs;
        if (per_row) { rs.cell = cell.data(); rs.slot = slot.data(); rs.kc = kcs.data(); rs.vc = vcs.data(); }
        const float* kc = per_row ? nullptr : kcs[0];
        const float* vc = per_row ? nullptr : vcs[0];
        if (q8) launch(n_chunk, T * HKV, 256, q36k::k_attn_part_gqa<q36k::KvQ8>, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx, scale, n_chunk, o.acc.data(), o.m.data(), o.l.data(), rs.cell, rs.slot, rs.kc, rs.vc);
        else launch(n_chunk, T * H, 256, k_attn_part<float>, (const float*) q.data(), kc, vc, (const int32_t*) &pos0, 0, H, HKV, ctx, scale, n_chunk, o.acc.data(), o.m.data(), o.l.data(), rs);
        return o;
    };
    const Out a = run(false), b = run(true);
    const bool ok = std::memcmp(a.acc.data(), b.acc.data(), a.acc.size() * 4) == 0 && std::memcmp(a.m.data(), b.m.data(), a.m.size() * 4) == 0 &&
                    std::memcmp(a.l.data(), b.l.data(), a.l.size() * 4) == 0;
    check(ok, what);
    return ok;
}

int main() {
    std::printf("k_attn_part_gqa against k_attn_part, bit for bit (CPU threads)\n");
    test_q8_store();
    {
        const char* kind = "[q8_0 cache] ";
        char buf[160];
        auto name = [&](const char* s) { std::snprintf(buf, sizeof buf, "%s%s", kind, s); return buf; };
        compare_q8(1, 600, 700, false, 21, name("decode, 1 token at cell 600, context 700 (2 full chunks + a partial one)"));
        compare_q8(1, 255, 512, false, 22, name("decode at cell 255: the last cell of chunk 0, context 512"));
        compare_q8(1, 0, 300, false, 23, name("the first token: one cell"));
        compare_q8(4, 300, 777, false, 24, name("a verify window of 4 tokens"));
        compare_q8(3, 50, 900, true, 25, name("3 rows on 3 sessions, per-row cells (RowSel)"));
        compare_q8(1, 699, 700, false, 26, name("the last cell of a 700-cell cache: the K read of the partial chunk is guarded"));
    }
    for (int half_cache = 0; half_cache < 2; ++half_cache) {
        const char* kind = half_cache ? "[f16 cache] " : "[f32 cache] ";
        char buf[160];
        auto name = [&](const char* s) { std::snprintf(buf, sizeof buf, "%s%s", kind, s); return buf; };
        const bool h = half_cache != 0;
        compare(1, 600, 700, false, h, 1, name("decode, 1 token at cell 600, context 700 (2 full chunks + a partial one)"));
        compare(1, 255, 512, false, h, 2, name("decode at cell 255: the last cell of chunk 0, context 512"));
        compare(1, 256, 512, false, h, 3, name("decode at cell 256: the first cell of chunk 1"));
        compare(1, 0, 300, false, h, 4, name("the first token: one cell"));
        compare(4, 300, 777, false, h, 5, name("a verify window of 4 tokens"));
        compare(3, 50, 900, true, h, 6, name("3 rows on 3 sessions, per-row cells (RowSel)"));
        compare(1, 699, 700, false, h, 7, name("the last cell of a 700-cell cache: the K read of the partial chunk is guarded"));
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
