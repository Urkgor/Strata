// g4_quant.hpp's dequantization against ggml's own (dequantize_row_*) on random blocks, every type the engine reads, unit by unit.  CPU only.
// Built with a llama.cpp checkout's ggml-base (see tools/gemma4/README.md); the repository's CMake links the `ggml` target.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "g4_quant.hpp"

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    std::mt19937 rng(5);
    const int types[] = {g4::T_F32, g4::T_F16, g4::T_BF16, g4::T_Q4_0, g4::T_Q4_1, g4::T_Q5_0, g4::T_Q5_1, g4::T_Q8_0, g4::T_Q2_K,
                         g4::T_Q3_K, g4::T_Q4_K, g4::T_Q5_K, g4::T_Q6_K, g4::T_IQ4_NL, g4::T_IQ4_XS};
    for (int t : types) {
        const int be = g4::block_elems(t), bb = g4::block_bytes(t);
        const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) t);
        if (t == g4::T_F32) continue;      // (ggml has no to_float for F32: the values are the floats)
        EXPECT(tr && tr->to_float, "type %s: ggml has no dequantizer", g4::type_name(t));
        if (!tr || !tr->to_float) continue;
        if (t != g4::T_F32 && t != g4::T_F16 && t != g4::T_BF16) {
            EXPECT((int) tr->type_size == bb && (int) tr->blck_size == be, "type %s: block %d / %d bytes, ggml says %d / %d", g4::type_name(t),
                   be, bb, (int) tr->blck_size, (int) tr->type_size);
        }
        // 3 rows of 3 superblocks (or blocks) of random bytes; the half scales are made finite (a random exponent could be inf / nan)
        const int n = 3 * (be == 32 ? 96 : 256);
        const long long nbytes = g4::row_bytes(t, n);
        for (int round = 0; round < 40; ++round) {
            std::vector<uint8_t> row((size_t) nbytes);
            for (auto& b : row) b = (uint8_t) rng();
            // keep every fp16 scale in the block finite and small: bytes of d / dmin / m are patched per format
            auto patch_half = [&](size_t at) { const uint16_t h = (uint16_t) (0x2c00 + (rng() & 0x03ff) + ((rng() & 1) << 15)); std::memcpy(&row[at], &h, 2); };
            const int nb = (int) (nbytes / bb);
            for (int b = 0; b < nb; ++b) {
                const size_t o = (size_t) b * (size_t) bb;
                switch (t) {
                    case g4::T_Q4_0: case g4::T_Q5_0: case g4::T_Q8_0: case g4::T_IQ4_NL: case g4::T_IQ4_XS: patch_half(o); break;
                    case g4::T_Q4_1: case g4::T_Q5_1: patch_half(o); patch_half(o + 2); break;
                    case g4::T_Q4_K: case g4::T_Q5_K: patch_half(o); patch_half(o + 2); break;
                    case g4::T_Q2_K: patch_half(o + 80); patch_half(o + 82); break;
                    case g4::T_Q3_K: patch_half(o + 108); break;
                    case g4::T_Q6_K: patch_half(o + 208); break;
                    case g4::T_F32: for (int j = 0; j < 32; ++j) { const float f = (float) ((int) (rng() % 2001) - 1000) / 250.f; std::memcpy(&row[o + 4 * j], &f, 4); } break;
                    case g4::T_F16: for (int j = 0; j < 32; ++j) patch_half(o + 2 * j); break;
                    case g4::T_BF16: for (int j = 0; j < 32; ++j) { const uint16_t h = (uint16_t) (0x3c00 + (rng() & 0x03ff) + ((rng() & 1) << 15)); std::memcpy(&row[o + 2 * j], &h, 2); } break;
                    default: break;
                }
            }
            std::vector<float> want((size_t) n), got((size_t) n);
            tr->to_float(row.data(), want.data(), n);
            g4::dequant_row(t, row.data(), n, got.data());
            size_t bad = 0, first = 0;
            for (int i = 0; i < n; ++i)
                if (std::memcmp(&want[(size_t) i], &got[(size_t) i], 4) != 0 && !(want[(size_t) i] == got[(size_t) i])) { if (!bad) first = (size_t) i; ++bad; }
            // the per-lane form the GPU kernels use must give the same value as the per-unit one
            size_t bad_e = 0;
            for (long long uu = 0; uu < n / 32; ++uu) {
                float unit[32];
                g4::dequant32_t(t, row.data(), uu, unit);
                for (int l = 0; l < 32; ++l) {
                    float e = 0.f;
                    switch (t) {
#define CASE(T) case g4::T: e = g4::dequant_elem<g4::T>(row.data(), uu, l); break;
                        CASE(T_F32) CASE(T_F16) CASE(T_BF16) CASE(T_Q4_0) CASE(T_Q4_1) CASE(T_Q5_0) CASE(T_Q5_1) CASE(T_Q8_0) CASE(T_Q2_K) CASE(T_Q3_K)
                        CASE(T_Q4_K) CASE(T_Q5_K) CASE(T_Q6_K) CASE(T_IQ4_NL) CASE(T_IQ4_XS)
#undef CASE
                        default: break;
                    }
                    if (!(e == unit[l])) ++bad_e;
                }
            }
            EXPECT(bad_e == 0, "%s round %d: dequant_elem differs from dequant32 in %zu values", g4::type_name(t), round, bad_e);
            EXPECT(bad == 0, "%s round %d: %zu of %d values differ, the first at %zu (ggml %g, mine %g)", g4::type_name(t), round, bad, n, first, want[first], got[first]);
            if (bad) break;
        }
    }
    std::printf(fails ? "%d FAILED\n" : "g4_quant_test: every type matches ggml\n", fails);
    return fails ? 1 : 0;
}
