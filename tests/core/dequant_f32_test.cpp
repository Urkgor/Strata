// dequantize_to_f32 (strata/kernels/cpu/native_expert.hpp: the small tensors the Qwen3.6 engine uploads as f32, such as ssm_alpha, when
// a file stores them as Q6_K / Q4_K / Q5_K / ...) against the values they were quantized from: ggml's own quantizer and dequantizer
// round-trip random data, so the error is the format's, not the code's.  No model, no GPU.
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

int main() {
    // 2048 x 32 values, the shape of ssm_alpha / ssm_beta (a multiple of every block size)
    constexpr int64_t kN = 2048 * 32;
    std::mt19937 rng(20261022);
    std::normal_distribution<float> gauss(0.0f, 0.05f);
    std::vector<float> x((size_t) kN);
    double energy = 0;
    for (auto& v : x) { v = gauss(rng); energy += (double) v * v; }
    struct Case { ggml_type type; const char* name; double max_rel_rms; };
    const Case cases[] = {{GGML_TYPE_Q8_0, "Q8_0", 0.01}, {GGML_TYPE_Q6_K, "Q6_K", 0.02}, {GGML_TYPE_Q5_K, "Q5_K", 0.05},
                          {GGML_TYPE_Q4_K, "Q4_K", 0.10}, {GGML_TYPE_Q5_0, "Q5_0", 0.06}, {GGML_TYPE_Q4_0, "Q4_0", 0.12}};
    for (const Case& c : cases) {
        const ggml_type_traits* t = ggml_get_type_traits(c.type);
        std::vector<uint8_t> q((size_t) (kN / t->blck_size) * t->type_size);
        t->from_float_ref(x.data(), q.data(), kN);
        std::vector<float> y((size_t) kN, 0.0f);
        const bool ok = strata::kernels::cpu::dequantize_to_f32((int) c.type, q.data(), y.data(), kN);
        double err = 0;
        for (int64_t i = 0; i < kN; ++i) err += (double) (x[(size_t) i] - y[(size_t) i]) * (x[(size_t) i] - y[(size_t) i]);
        const double rel = std::sqrt(err / energy);
        char what[96];
        std::snprintf(what, sizeof what, "%s: widened, relative RMS error %.4f (limit %.2f)", c.name, rel, c.max_rel_rms);
        check(ok && rel < c.max_rel_rms, what);
        // the same values as ggml's own to_float, bit for bit
        std::vector<float> w((size_t) kN);
        t->to_float(q.data(), w.data(), kN);
        bool same = true;
        for (int64_t i = 0; i < kN && same; ++i) same = w[(size_t) i] == y[(size_t) i];
        std::snprintf(what, sizeof what, "%s: identical to ggml's to_float", c.name);
        check(same, what);
    }
    float out[512];
    std::vector<uint8_t> junk(4096, 0);
    check(!strata::kernels::cpu::dequantize_to_f32((int) GGML_TYPE_Q6_K, junk.data(), out, 100),
          "a count that is not a multiple of the block size is refused");
    check(!strata::kernels::cpu::dequantize_to_f32(-1, junk.data(), out, 256) &&
              !strata::kernels::cpu::dequantize_to_f32((int) GGML_TYPE_COUNT + 5, junk.data(), out, 256),
          "a type ggml does not know is refused");
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
