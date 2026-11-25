#include "i2_cpu_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../gemma4/g4_quant.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define I2_X86 1
#endif

namespace i2cpu {

namespace {

float hsum(const float* v, int n) { float s = 0.f; for (int i = 0; i < n; ++i) s += v[i]; return s; }

#ifdef I2_X86
__attribute__((target("avx2,fma"))) inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma"))) float dot_iq4nl_avx2(const uint8_t* row, const float* x, int n) {
    alignas(16) static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m128i lut = _mm_load_si128((const __m128i*) kv), m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* p = row + (size_t) b * 18;
        const float d = g4::half_to_float((uint16_t) (p[0] | (p[1] << 8)));
        const __m128i q = _mm_loadu_si128((const __m128i*) (p + 2));
        const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
        const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
        const float* xp = x + (size_t) b * 32;
        __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_loadu_ps(xp));
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_loadu_ps(xp + 8), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), _mm256_loadu_ps(xp + 16), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), _mm256_loadu_ps(xp + 24), s);
        acc = _mm256_fmadd_ps(s, _mm256_set1_ps(d), acc);
    }
    return hsum256(acc);
}

__attribute__((target("avx2,fma"))) float dot_nvfp4_avx2(const uint8_t* row, const float* x, int n) {
    alignas(16) static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const __m128i lut = _mm_load_si128((const __m128i*) kv), m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 64; ++b) {
        const uint8_t* p = row + (size_t) b * 36;
        const float* xp = x + (size_t) b * 64;
        for (int s = 0; s < 4; ++s) {                       // a sub-block: 16 values, 8 bytes (the low nibbles the first 8 values, the high nibbles the last 8)
            const __m128i q = _mm_loadl_epi64((const __m128i*) (p + 4 + s * 8));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
            __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_loadu_ps(xp + s * 16));
            t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), _mm256_loadu_ps(xp + s * 16 + 8), t);
            acc = _mm256_fmadd_ps(t, _mm256_set1_ps(g4::ue4m3_half(p[s])), acc);
        }
    }
    return hsum256(acc);
}

__attribute__((target("avx2,fma"))) float dot_q8_0_avx2(const uint8_t* row, const float* x, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* p = row + (size_t) b * 34;
        const float d = g4::half_to_float((uint16_t) (p[0] | (p[1] << 8)));
        const float* xp = x + (size_t) b * 32;
        __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (p + 2)))), _mm256_loadu_ps(xp));
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (p + 10)))), _mm256_loadu_ps(xp + 8), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (p + 18)))), _mm256_loadu_ps(xp + 16), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (p + 26)))), _mm256_loadu_ps(xp + 24), s);
        acc = _mm256_fmadd_ps(s, _mm256_set1_ps(d), acc);
    }
    return hsum256(acc);
}

// ---- the same dots against int8 activations (blocks of 32 values, one scale each): 32 products in two instructions instead of twelve
__attribute__((target("avx2,fma"))) inline __m256i dot_i8x32(__m256i w, __m256i x) {
    const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));     // |w| * (x with w's sign): two products a lane, no overflow (<= 2 * 127 * 128)
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}

__attribute__((target("avx2,fma,f16c"))) inline float half_at(const uint8_t* p) { return _cvtsh_ss((unsigned short) (p[0] | (p[1] << 8))); }

__attribute__((target("avx2,fma,f16c"))) float dot_q8_0_i8_avx2(const uint8_t* row, const int8_t* xq, const float* xd, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* p = row + (size_t) b * 34;
        const float d = half_at(p) * xd[b];
        const __m256i w = _mm256_loadu_si256((const __m256i*) (p + 2));       // (unaligned: 34-byte blocks)
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8x32(w, _mm256_loadu_si256((const __m256i*) (xq + (size_t) b * 32)))), _mm256_set1_ps(d), acc);
    }
    return hsum256(acc);
}

__attribute__((target("avx2,fma,f16c"))) float dot_iq4nl_i8_avx2(const uint8_t* row, const int8_t* xq, const float* xd, int n) {
    alignas(16) static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m128i lut = _mm_load_si128((const __m128i*) kv), m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* p = row + (size_t) b * 18;
        const float d = half_at(p) * xd[b];
        const __m128i q = _mm_loadu_si128((const __m128i*) (p + 2));
        const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
        const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
        const __m256i w = _mm256_set_m128i(hi, lo);                          // values 0..15 from the low nibbles, 16..31 from the high ones
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8x32(w, _mm256_loadu_si256((const __m256i*) (xq + (size_t) b * 32)))), _mm256_set1_ps(d), acc);
    }
    return hsum256(acc);
}

/// the NVFP4 sub-block scales by byte (the float of g4::ue4m3_half)
const float* ue4m3_table() {
    static const std::vector<float> t = [] { std::vector<float> v(256); for (int i = 0; i < 256; ++i) v[(size_t) i] = g4::ue4m3_half(i); return v; }();
    return t.data();
}

// A block of 64 values is 4 sub-blocks of 16 (a scale byte each) over 32 data bytes: sub-block k has its first 8 values in the low nibbles of bytes 8k..8k+7 and its last 8 in the high nibbles.
// One 32-byte load unpacks to (sub-blocks 0 | 2) and (1 | 3) in the two 128-bit lanes; the activations of sub-blocks 0 and 1 are one block of 32 (scale dx0), 2 and 3 the next (dx1).
__attribute__((target("avx2,fma,f16c"))) float dot_nvfp4_i8_avx2(const uint8_t* row, const int8_t* xq, const float* xd, int n) {
    alignas(16) static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) kv)), m4 = _mm256_set1_epi8(0x0F);
    const float* ue = ue4m3_table();
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 64; ++b) {
        const uint8_t* p = row + (size_t) b * 36;
        const __m256i q = _mm256_loadu_si256((const __m256i*) (p + 4));
        const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));
        const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
        const __m256i wa = _mm256_unpacklo_epi64(lo, hi);                    // sub-blocks 0 | 2
        const __m256i wb = _mm256_unpackhi_epi64(lo, hi);                    // sub-blocks 1 | 3
        const int8_t* x0 = xq + (size_t) b * 64;
        const __m256i xa = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) x0)), _mm_loadu_si128((const __m128i*) (x0 + 32)), 1);        // x[0..16) | x[32..48)
        const __m256i xb = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) (x0 + 16))), _mm_loadu_si128((const __m128i*) (x0 + 48)), 1);   // x[16..32) | x[48..64)
        const float dx0 = xd[b * 2], dx1 = xd[b * 2 + 1];
        const __m256 sa = _mm256_set_m128(_mm_set1_ps(ue[p[2]] * dx1), _mm_set1_ps(ue[p[0]] * dx0));
        const __m256 sb = _mm256_set_m128(_mm_set1_ps(ue[p[3]] * dx1), _mm_set1_ps(ue[p[1]] * dx0));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8x32(wa, xa)), sa, acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8x32(wb, xb)), sb, acc);
    }
    return hsum256(acc);
}
#endif

}  // namespace

bool have_avx2() {
#ifdef I2_X86
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return ok;
#else
    return false;
#endif
}

float dot_row_generic(int type, const uint8_t* row, const float* x, int n) {
    float tmp[32], part[8];
    double s = 0.0;
    for (int u = 0; u < n / 32; ++u) {
        g4::dequant32_t(type, row, (long long) u, tmp);
        for (int g = 0; g < 4; ++g) {
            for (int j = 0; j < 8; ++j) part[j] = tmp[g * 8 + j] * x[u * 32 + g * 8 + j];
            s += hsum(part, 8);
        }
    }
    return (float) s;
}

float dot_row(int type, const uint8_t* row, const float* x, int n) {
#ifdef I2_X86
    if (have_avx2()) {
        if (type == g4::T_IQ4_NL) return dot_iq4nl_avx2(row, x, n);
        if (type == g4::T_Q8_0) return dot_q8_0_avx2(row, x, n);
        if (type == g4::T_NVFP4) return dot_nvfp4_avx2(row, x, n);
    }
#endif
    return dot_row_generic(type, row, x, n);
}

bool int8_type(int type) {
#ifdef I2_X86
    return have_avx2() && (type == g4::T_IQ4_NL || type == g4::T_Q8_0 || type == g4::T_NVFP4);
#else
    (void) type;
    return false;
#endif
}

void quantize_i8(const float* x, int n, int8_t* q, float* d) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float sc = amax / 127.f, inv = sc > 0.f ? 1.f / sc : 0.f;
        d[b] = sc;
        for (int i = 0; i < 32; ++i) q[b * 32 + i] = (int8_t) std::lrintf(std::max(-127.f, std::min(127.f, x[b * 32 + i] * inv)));
    }
}

float dot_row_i8(int type, const uint8_t* row, const int8_t* xq, const float* xd, int n) {
#ifdef I2_X86
    if (type == g4::T_IQ4_NL) return dot_iq4nl_i8_avx2(row, xq, xd, n);
    if (type == g4::T_Q8_0) return dot_q8_0_i8_avx2(row, xq, xd, n);
    if (type == g4::T_NVFP4) return dot_nvfp4_i8_avx2(row, xq, xd, n);
#endif
    (void) row; (void) xq; (void) xd; (void) n; (void) type;
    return 0.f;
}

// ------------------------------------------------------------------------------------------------------------------------ the pool
Pool::Pool(int threads) {
    for (int i = 1; i < std::max(1, threads); ++i) workers_.emplace_back([this] { worker(); });
}

Pool::~Pool() {
    stop_ = true;
    {
        std::lock_guard<std::mutex> lk(mu_);
        gen_.fetch_add(1);
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void Pool::worker() {
    uint64_t seen = 0;
    for (;;) {
        // wait for a new generation: a short spin (the phases of a step follow each other closely), then sleep
        int spins = 0;
        while (gen_.load(std::memory_order_acquire) == seen && !stop_.load()) {
            if (++spins < 200000) {
#ifdef I2_X86
                _mm_pause();
#endif
                if ((spins & 255) == 0) std::this_thread::yield();      // (on a machine with no spare core this lets the thread that launches the GPU's kernels run)
                continue;
            }
            std::unique_lock<std::mutex> lk(mu_);
            sleepers_.fetch_add(1);
            cv_.wait(lk, [&] { return gen_.load() != seen || stop_.load(); });
            sleepers_.fetch_sub(1);
        }
        if (stop_.load()) return;
        seen = gen_.load(std::memory_order_acquire);
        const auto* fn = fn_;
        const size_t total = total_, chunk = chunk_;
        for (;;) {
            const size_t b = next_.fetch_add(chunk);
            if (b >= total) break;
            (*fn)(b, std::min(total, b + chunk));
        }
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void Pool::parallel_for(size_t total, size_t chunk, const std::function<void(size_t, size_t)>& fn) {
    if (workers_.empty() || total <= chunk) {
        for (size_t b = 0; b < total; b += chunk) fn(b, std::min(total, b + chunk));
        return;
    }
    fn_ = &fn;
    total_ = total;
    chunk_ = chunk;
    next_.store(0);
    done_.store(0);
    {
        std::lock_guard<std::mutex> lk(mu_);
        gen_.fetch_add(1, std::memory_order_release);
    }
    if (sleepers_.load() > 0) cv_.notify_all();
    for (;;) {
        const size_t b = next_.fetch_add(chunk);
        if (b >= total) break;
        fn(b, std::min(total, b + chunk));
    }
    // every chunk is taken; every worker answers each generation (one that wakes late finds next_ past the end), so wait for all of them
    while (done_.load(std::memory_order_acquire) < (int) workers_.size()) {
#ifdef I2_X86
        _mm_pause();
#endif
    }
    fn_ = nullptr;
}

void Pool::run(const float* x, int E, int F, float beta, float lin, const Job* jobs, int n) {
    if (n <= 0) return;
    const size_t rows2 = (size_t) E;
    bool use_i8 = int8_ && E % 32 == 0 && F % 32 == 0;
    for (int j = 0; j < n && use_i8; ++j)
        use_i8 = int8_type(jobs[j].gu_type) && int8_type(jobs[j].dn_type) && (jobs[j].gu_type != g4::T_NVFP4 || E % 64 == 0) && (jobs[j].dn_type != g4::T_NVFP4 || F % 64 == 0);
    const size_t nbx = (size_t) E / 32, nbf = (size_t) F / 32;
    if (use_i8) {
        // gate+up, the activation and its quantization in one phase (a task: 32 columns of one job: 64 rows, 32 products, one block of the next input), then the down rows
        xq_.resize((size_t) E);
        xd_.resize(nbx);
        quantize_i8(x, E, xq_.data(), xd_.data());
        aq_.resize((size_t) n * (size_t) F);
        ad_.resize((size_t) n * nbf);
        parallel_for((size_t) n * nbf, 1, [&](size_t b, size_t e) {
            for (size_t t = b; t < e; ++t) {
                const size_t j = t / nbf, c0 = (t % nbf) * 32;
                const Job& jb = jobs[j];
                float a[32];
                for (size_t k = 0; k < 32; ++k) {
                    const size_t c = c0 + k;
                    const float g = jb.gu_scale * dot_row_i8(jb.gu_type, jb.gu + c * jb.gu_row_bytes, xq_.data(), xd_.data(), E);
                    const float u = jb.gu_scale * dot_row_i8(jb.gu_type, jb.gu + ((size_t) F + c) * jb.gu_row_bytes, xq_.data(), xd_.data(), E);
                    const float sg = 1.0f / (1.0f + expf(-g));
                    a[k] = (beta * tanhf(g / beta) * sg) * (lin * tanhf(u / lin));
                }
                quantize_i8(a, 32, aq_.data() + j * (size_t) F + c0, ad_.data() + j * nbf + c0 / 32);
            }
        });
        parallel_for((size_t) n * rows2, 16, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) {
                const size_t j = i / rows2, r = i % rows2;
                jobs[j].y[r] = dot_row_i8(jobs[j].dn_type, jobs[j].dn + r * jobs[j].dn_row_bytes, aq_.data() + j * (size_t) F, ad_.data() + j * nbf, F);
            }
        });
        return;
    }
    // floats: gate+up and the activation in one phase (32 columns of one job a task), then the down rows
    act_.resize((size_t) n * (size_t) F);
    parallel_for((size_t) n * nbf, 1, [&](size_t b, size_t e) {
        for (size_t t = b; t < e; ++t) {
            const size_t j = t / nbf, c0 = (t % nbf) * 32;
            const Job& jb = jobs[j];
            for (size_t k = 0; k < 32; ++k) {
                const size_t c = c0 + k;
                const float g = jb.gu_scale * dot_row(jb.gu_type, jb.gu + c * jb.gu_row_bytes, x, E);
                const float u = jb.gu_scale * dot_row(jb.gu_type, jb.gu + ((size_t) F + c) * jb.gu_row_bytes, x, E);
                const float sg = 1.0f / (1.0f + expf(-g));
                act_[j * (size_t) F + c] = (beta * tanhf(g / beta) * sg) * (lin * tanhf(u / lin));
            }
        }
    });
    parallel_for((size_t) n * rows2, 16, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            const size_t j = i / rows2, r = i % rows2;
            jobs[j].y[r] = dot_row(jobs[j].dn_type, jobs[j].dn + r * jobs[j].dn_row_bytes, act_.data() + j * (size_t) F, F);
        }
    });
}

}  // namespace i2cpu
