// The CPU expert pool (src/inkling2/i2_cpu_pool.cpp): the AVX2 rows against the portable path, and a whole expert through the pool against a scalar evaluation.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "i2_cpu_pool.hpp"
#include "../gemma4/g4_quant.hpp"

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); } else std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

static void fill(int type, std::vector<uint8_t>& row, std::mt19937& rng) {
    for (auto& b : row) b = (uint8_t) rng();
    const int bb = g4::block_bytes(type) == 128 ? 128 : g4::block_bytes(type);
    // sane scales: the half at the start of every block
    for (size_t o = 0; o + 2 <= row.size() && (type == g4::T_IQ4_NL || type == g4::T_Q8_0); o += (size_t) bb) {
        const float d = 0.001f + 0.01f * (float) (rng() % 100) / 100.f;
        const uint16_t h = (uint16_t) std::lrintf(d * 1024.f * 16.f);   // not exact IEEE, but a finite positive half-ish value: only both paths' agreement matters
        row[o] = (uint8_t) (h & 0xff);
        row[o + 1] = (uint8_t) (0x20 | ((h >> 8) & 0x03));
    }
}

int main() {
    std::mt19937 rng(7);
    std::printf("avx2: %s\n", i2cpu::have_avx2() ? "yes" : "no (the portable path only)");
    for (int type : {g4::T_IQ4_NL, g4::T_Q8_0, g4::T_NVFP4, g4::T_F16}) {
        const int n = 2304;
        const size_t rb = (size_t) g4::row_bytes(type, n);
        std::vector<uint8_t> row(rb);
        std::vector<float> x((size_t) n);
        double worst = 0;
        for (int rep = 0; rep < 20; ++rep) {
            fill(type, row, rng);
            if (type == g4::T_F16) for (size_t i = 0; i + 1 < row.size(); i += 2) row[i + 1] = (uint8_t) ((row[i + 1] & 0x83) | 0x20);   // small finite halves
            for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 500.f;
            const float a = i2cpu::dot_row(type, row.data(), x.data(), n), b = i2cpu::dot_row_generic(type, row.data(), x.data(), n);
            worst = std::max(worst, (double) std::fabs(a - b) / (std::fabs(b) + 1e-3));
        }
        EXPECT(worst < 1e-4, "row dot of type %d against the portable path: worst relative difference %.2e", type, worst);
    }
    {   // an expert: IQ4_NL gate_up [2F][E], Q8_0 down [E][F], through the pool with 3 threads
        const int E = 256, F = 96;
        const size_t rb1 = (size_t) E / 32 * 18, rb2 = (size_t) F / 32 * 34;
        std::vector<uint8_t> gu(rb1 * 2 * (size_t) F), dn(rb2 * (size_t) E);
        std::vector<float> x((size_t) E);
        i2cpu::Pool pool(3);
        for (int rep = 0; rep < 5; ++rep) {
            for (size_t r = 0; r < 2 * (size_t) F; ++r) { std::vector<uint8_t> t(rb1); fill(g4::T_IQ4_NL, t, rng); std::copy(t.begin(), t.end(), gu.begin() + (long) (r * rb1)); }
            for (size_t r = 0; r < (size_t) E; ++r) { std::vector<uint8_t> t(rb2); fill(g4::T_Q8_0, t, rng); std::copy(t.begin(), t.end(), dn.begin() + (long) (r * rb2)); }
            for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
            std::vector<float> y(E), want(E), u(2 * (size_t) F), act(F);
            i2cpu::Job job{gu.data(), g4::T_IQ4_NL, rb1, dn.data(), g4::T_Q8_0, rb2, y.data()};
            pool.run(x.data(), E, F, 4.f, 25.f, &job, 1);
            for (int r = 0; r < 2 * F; ++r) u[(size_t) r] = i2cpu::dot_row_generic(g4::T_IQ4_NL, gu.data() + (size_t) r * rb1, x.data(), E);
            for (int j = 0; j < F; ++j) {
                const float g = u[(size_t) j], up = u[(size_t) F + (size_t) j];
                act[(size_t) j] = (4.f * tanhf(g / 4.f) / (1.f + expf(-g))) * (25.f * tanhf(up / 25.f));
            }
            double d2 = 0, s2 = 0;
            for (int r = 0; r < E; ++r) {
                want[(size_t) r] = i2cpu::dot_row_generic(g4::T_Q8_0, dn.data() + (size_t) r * rb2, act.data(), F);
                d2 += (double) (y[(size_t) r] - want[(size_t) r]) * (y[(size_t) r] - want[(size_t) r]);
                s2 += (double) want[(size_t) r] * want[(size_t) r];
            }
            EXPECT(std::sqrt(d2 / std::max(s2, 1e-30)) < 1e-4, "pool expert %d: relative error %.2e", rep, std::sqrt(d2 / std::max(s2, 1e-30)));
        }
    }
    std::printf(fails ? "%d FAILED\n" : "i2_cpu_pool_test: all pass\n", fails);
    return fails ? 1 : 0;
}
