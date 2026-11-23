// k_attn_pf (a prompt chunk's attention, 8 queries a block) against k_attn on the same random inputs, on the CPU emulation of CUDA.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../gemma4/g4_cuda_compat.hpp"
#include "../gemma4/g4_quant.hpp"
#include "i2_kernels.cuh"

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); } else std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

template <class T> static T* dev(const std::vector<T>& v) { T* d; cudaMalloc((void**) &d, v.size() * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }

int main() {
    std::mt19937 rng(5);
    auto rnd = [&](float s) { return s * (float) ((int) (rng() % 2001) - 1000) / 1000.f; };
    constexpr int NU = 4, HD = 128;
    const int n_head = 6, n_kv = 2, n_rel = 16, extent = 40, ring = 96, T = 21, pos0 = 33;
    for (int kvq = 0; kvq <= 2; ++kvq)
        for (int window : {0, 29}) {
            const size_t cells = (size_t) n_kv * (size_t) ring;
            std::vector<uint16_t> kc, vc, ks, vs;
            std::vector<uint8_t> kq, vq;
            if (kvq == 0) { kc.resize(cells * HD); vc.resize(cells * HD); for (auto& x : kc) x = g4::float_to_half(rnd(1.5f)); for (auto& x : vc) x = g4::float_to_half(rnd(1.5f)); }
            else {
                const size_t per = kvq == 1 ? HD : HD / 2;
                kq.resize(cells * per); vq.resize(cells * per); ks.resize(cells * NU); vs.resize(cells * NU);
                for (auto& x : kq) x = (uint8_t) rng(); for (auto& x : vq) x = (uint8_t) rng();
                for (auto& x : ks) x = g4::float_to_half(0.01f + 0.01f * (rng() % 10)); for (auto& x : vs) x = g4::float_to_half(0.01f + 0.01f * (rng() % 10));
            }
            std::vector<float> q((size_t) T * n_head * HD), r((size_t) T * n_head * n_rel), rb((size_t) extent * n_rel), sink((size_t) T * n_head), gate((size_t) T * n_head * HD);
            for (auto& x : q) x = rnd(1.f); for (auto& x : r) x = rnd(1.f); for (auto& x : rb) x = rnd(0.5f); for (auto& x : sink) x = rnd(1.f); for (auto& x : gate) x = rnd(2.f);
            auto dq = dev(q); auto dr = dev(r); auto drb = dev(rb); auto dsk = dev(sink); auto dg = dev(gate);
            uint16_t *dkc = kc.empty() ? nullptr : dev(kc), *dvc = vc.empty() ? nullptr : dev(vc), *dks = ks.empty() ? nullptr : dev(ks), *dvs = vs.empty() ? nullptr : dev(vs);
            uint8_t *dkq = kq.empty() ? nullptr : dev(kq), *dvq = vq.empty() ? nullptr : dev(vq);
            float *o1, *o2;
            cudaMalloc((void**) &o1, q.size() * 4); cudaMalloc((void**) &o2, q.size() * 4);
            float *pm, *pl, *pacc; cudaMalloc((void**) &pm, 64); cudaMalloc((void**) &pl, 64); cudaMalloc((void**) &pacc, 64);
            auto run = [&](auto tag) {
                constexpr int KV = decltype(tag)::value;
                G4_LAUNCH((i2k::k_attn<NU, KV>), dim3((unsigned) T, (unsigned) n_head, 1u), dim3(128), 0, dq, dr, drb, dsk, dg, (const uint16_t*) dkc, (const uint16_t*) dvc, (const uint8_t*) dkq, (const uint8_t*) dvq,
                          (const uint16_t*) dks, (const uint16_t*) dvs, o1, n_head, n_kv, n_rel, extent, pos0, ring, window, 1, pm, pl, pacc);
                G4_LAUNCH((i2k::k_attn_pf<NU, KV>), dim3((unsigned) ((T + 7) / 8), (unsigned) n_head), dim3(256), 0, dq, dr, drb, dsk, dg, (const uint16_t*) dkc, (const uint16_t*) dvc, (const uint8_t*) dkq, (const uint8_t*) dvq,
                          (const uint16_t*) dks, (const uint16_t*) dvs, o2, n_head, n_kv, n_rel, extent, pos0, ring, window, T);
            };
            if (kvq == 0) run(std::integral_constant<int, 0>{}); else if (kvq == 1) run(std::integral_constant<int, 1>{}); else run(std::integral_constant<int, 2>{});
            std::vector<float> a(q.size()), b(q.size());
            cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost);
            double d2 = 0, s2 = 0;
            for (size_t i = 0; i < a.size(); ++i) { d2 += (double) (a[i] - b[i]) * (a[i] - b[i]); s2 += (double) a[i] * a[i]; }
            EXPECT(std::sqrt(d2 / std::max(s2, 1e-30)) < 1e-4, "k_attn_pf against k_attn, cache %s, %s: relative difference %.2e", kvq == 0 ? "halves" : kvq == 1 ? "q8_0" : "4 bits", window ? "window 29" : "global", std::sqrt(d2 / std::max(s2, 1e-30)));
        }
    std::printf(fails ? "%d FAILED\n" : "i2_attn_test: all pass\n", fails);
    return fails ? 1 : 0;
}
