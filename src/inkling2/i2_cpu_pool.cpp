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
#endif

}  // namespace

bool have_avx2() {
#ifdef I2_X86
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
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
    std::vector<float> u((size_t) n * 2 * (size_t) F), act((size_t) n * (size_t) F);
    const size_t rows1 = (size_t) 2 * (size_t) F, rows2 = (size_t) E;
    parallel_for((size_t) n * rows1, 16, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            const size_t j = i / rows1, r = i % rows1;
            u[i] = jobs[j].gu_scale * dot_row(jobs[j].gu_type, jobs[j].gu + r * jobs[j].gu_row_bytes, x, E);
        }
    });
    parallel_for((size_t) n * (size_t) F, 256, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            const size_t j = i / (size_t) F, c = i % (size_t) F;
            const float g = u[j * rows1 + c], up = u[j * rows1 + (size_t) F + c];
            const float sg = 1.0f / (1.0f + expf(-g));
            act[i] = (beta * tanhf(g / beta) * sg) * (lin * tanhf(up / lin));
        }
    });
    parallel_for((size_t) n * rows2, 16, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            const size_t j = i / rows2, r = i % rows2;
            jobs[j].y[r] = dot_row(jobs[j].dn_type, jobs[j].dn + r * jobs[j].dn_row_bytes, act.data() + j * (size_t) F, F);
        }
    });
}

}  // namespace i2cpu
