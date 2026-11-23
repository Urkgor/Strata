// The integer decode kernels (k_quant_x, k_mv in i2_kernels.cuh) on the CPU emulation of CUDA against the float dequantization of the same weights.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../gemma4/g4_cuda_compat.hpp"
#include "../gemma4/g4_quant.hpp"
#include "i2_kernels.cuh"

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); } else std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

int main() {
    std::mt19937 rng(11);
    for (int type : {g4::T_Q8_0, g4::T_NVFP4}) {
        const int n_in = 2304, n_out = 40, n_exp = 3, ne = 8;       // 3 experts' worth of rows; 8 entries read experts 2, 0, 1, 2, ... with their own (or one shared) activations
        const size_t rb = (size_t) g4::row_bytes(type, n_in), eb = rb * (size_t) n_out;
        std::vector<uint8_t> w(eb * (size_t) n_exp);
        for (auto& b : w) b = (uint8_t) rng();
        const int bb = g4::block_bytes(type), be = g4::block_elems(type);
        for (size_t o = 0; o + (size_t) bb <= w.size(); o += (size_t) bb) {         // sane scales
            if (type == g4::T_Q8_0) { const uint16_t h = (uint16_t) (0x2000 | (rng() & 0x3ff)); w[o] = (uint8_t) (h & 0xff); w[o + 1] = (uint8_t) (h >> 8); }
            else for (int s = 0; s < 4; ++s) w[o + (size_t) s] = (uint8_t) (0x20 + (rng() % 40));
        }
        (void) be;
        for (int shared = 0; shared < 2; ++shared) {
            std::vector<float> x((size_t) ne * (size_t) n_in);
            for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 500.f;
            std::vector<int> ge(ne);
            for (int k = 0; k < ne; ++k) ge[(size_t) k] = (k * 2 + 2) % n_exp;
            uint8_t* dw; float *dx, *dxd, *dy; int8_t* dxq; int* dge;
            cudaMalloc((void**) &dw, w.size()); cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
            cudaMalloc((void**) &dx, x.size() * 4); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &dxq, x.size()); cudaMalloc((void**) &dxd, x.size() / 8 + 64);
            cudaMalloc((void**) &dy, (size_t) ne * (size_t) n_out * 4);
            cudaMalloc((void**) &dge, (size_t) ne * 4); cudaMemcpy(dge, ge.data(), (size_t) ne * 4, cudaMemcpyHostToDevice);
            const int rows = shared ? 1 : ne;
            G4_LAUNCH(i2k::k_quant_x, dim3((unsigned) ((n_in / 32 + 7) / 8), (unsigned) rows), dim3(256), 0, dx, n_in, dxq, dxd);
            auto run = [&](auto tag) {
                constexpr int TY = decltype(tag)::value;
                G4_LAUNCH(G4_K(i2k::k_mv<TY>), dim3((unsigned) ((n_out + 7) / 8), (unsigned) ne), dim3(256), 0, dw, (long long) eb, (long long) rb, n_in, n_out, (const int8_t*) dxq, (const float*) dxd,
                          (long long) (shared ? 0 : n_in), (const int*) dge, dy, n_out, 0);
            };
            if (type == g4::T_Q8_0) run(std::integral_constant<int, g4::T_Q8_0>{}); else run(std::integral_constant<int, g4::T_NVFP4>{});
            std::vector<float> y((size_t) ne * (size_t) n_out);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            double d2 = 0, s2 = 0;
            std::vector<float> wrow((size_t) n_in);
            for (int k = 0; k < ne; ++k)
                for (int r = 0; r < n_out; ++r) {
                    g4::dequant_row(type, w.data() + (size_t) ge[(size_t) k] * eb + (size_t) r * rb, n_in, wrow.data());
                    const float* xr = x.data() + (shared ? 0 : (size_t) k * (size_t) n_in);
                    double dot = 0;
                    for (int i = 0; i < n_in; ++i) dot += (double) wrow[(size_t) i] * xr[i];
                    const double got = y[(size_t) k * (size_t) n_out + (size_t) r];
                    d2 += (got - dot) * (got - dot);
                    s2 += dot * dot;
                }
            const double rel = std::sqrt(d2 / std::max(s2, 1e-30));
            EXPECT(rel < 1.5e-2, "type %d, %s activations: rms error of the integer kernel against the float dot %.2e (activations quantized to 8 bits: about 4e-3)", type, shared ? "one shared set of" : "a set per entry of", rel);
            cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dxd); cudaFree(dy); cudaFree(dge);
        }
    }
    // reading a prompt: k_mmq, a dense matrix over a ragged number of tokens, and expert groups with gathered rows
    for (int type : {g4::T_Q8_0, g4::T_NVFP4}) {
        const int n_in = 2304, n_out = 24, ntok = 37, n_exp = 3;
        const size_t rb = (size_t) g4::row_bytes(type, n_in), eb = rb * (size_t) n_out;
        std::vector<uint8_t> w(eb * (size_t) n_exp);
        for (auto& b : w) b = (uint8_t) rng();
        const int bb = g4::block_bytes(type);
        for (size_t o = 0; o + (size_t) bb <= w.size(); o += (size_t) bb) {
            if (type == g4::T_Q8_0) { const uint16_t h = (uint16_t) (0x2000 | (rng() & 0x3ff)); w[o] = (uint8_t) (h & 0xff); w[o + 1] = (uint8_t) (h >> 8); }
            else for (int s = 0; s < 4; ++s) w[o + (size_t) s] = (uint8_t) (0x20 + (rng() % 40));
        }
        std::vector<float> x((size_t) ntok * (size_t) n_in);
        for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 500.f;
        // experts: 3 groups: expert 2 reads the tokens 5, 9, 30; expert 0 the tokens 0 .. 15; expert 1 the token 36
        std::vector<int> ge = {2, 0, 1}, gf = {0, 3, 19}, gn = {3, 16, 1}, ein, eout;
        for (int t : {5, 9, 30}) ein.push_back(t);
        for (int t = 0; t < 16; ++t) ein.push_back(t);
        ein.push_back(36);
        for (int i = 0; i < (int) ein.size(); ++i) eout.push_back(ntok - 1 - i);                // (the outputs go to other rows than the inputs came from)
        uint8_t* dw; float *dx, *dxd, *dy; int8_t* dxq; int *dge, *dgf, *dgn, *dei, *deo;
        auto up = [&](auto*& d, const auto& v) { cudaMalloc((void**) &d, v.size() * sizeof(v[0])); cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice); };
        up(dw, w); up(dx, x); up(dge, ge); up(dgf, gf); up(dgn, gn); up(dei, ein); up(deo, eout);
        cudaMalloc((void**) &dxq, x.size()); cudaMalloc((void**) &dxd, x.size() / 8 + 64); cudaMalloc((void**) &dy, (size_t) ntok * (size_t) n_out * 4);
        G4_LAUNCH(i2k::k_quant_x, dim3((unsigned) ((n_in / 32 + 7) / 8), (unsigned) ntok), dim3(256), 0, dx, n_in, dxq, dxd);
        auto run = [&](auto tag, bool dense) {
            constexpr int TY = decltype(tag)::value;
            if (dense) G4_LAUNCH((i2k::k_mmq<TY, 16>), dim3((unsigned) ((n_out + 15) / 16), (unsigned) ((ntok + 15) / 16)), dim3(256), 0, dw, (long long) eb, (long long) rb, n_in, n_out, (const int8_t*) dxq,
                                 (const float*) dxd, (const int*) nullptr, (const int*) nullptr, (const int*) nullptr, (const int*) nullptr, (const int*) nullptr, ntok, dy, n_out, 0);
            else G4_LAUNCH((i2k::k_mmq<TY, 16>), dim3((unsigned) ((n_out + 15) / 16), 3u), dim3(256), 0, dw, (long long) eb, (long long) rb, n_in, n_out, (const int8_t*) dxq, (const float*) dxd,
                           (const int*) dge, (const int*) dgf, (const int*) dgn, (const int*) dei, (const int*) deo, ntok, dy, n_out, 0);
        };
        std::vector<float> y((size_t) ntok * (size_t) n_out), wrow((size_t) n_in);
        for (int dense = 1; dense >= 0; --dense) {
            cudaMemset(dy, 0, y.size() * 4);
            if (type == g4::T_Q8_0) run(std::integral_constant<int, g4::T_Q8_0>{}, dense); else run(std::integral_constant<int, g4::T_NVFP4>{}, dense);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            double d2 = 0, s2 = 0;
            const int n_ent = dense ? ntok : (int) ein.size();
            for (int k = 0; k < n_ent; ++k) {
                const int xr = dense ? k : ein[(size_t) k], yr = dense ? k : eout[(size_t) k];
                const int e = dense ? 0 : (k < 3 ? 2 : k < 19 ? 0 : 1);
                for (int r = 0; r < n_out; ++r) {
                    g4::dequant_row(type, w.data() + (size_t) e * eb + (size_t) r * rb, n_in, wrow.data());
                    double dot = 0;
                    for (int i = 0; i < n_in; ++i) dot += (double) wrow[(size_t) i] * x[(size_t) xr * (size_t) n_in + (size_t) i];
                    const double got = y[(size_t) yr * (size_t) n_out + (size_t) r];
                    d2 += (got - dot) * (got - dot);
                    s2 += dot * dot;
                }
            }
            const double rel = std::sqrt(d2 / std::max(s2, 1e-30));
            EXPECT(rel < 1.5e-2, "type %d, k_mmq %s (%d entries): rms error against the float dot %.2e", type, dense ? "dense over 37 tokens" : "expert groups with gathered rows", n_ent, rel);
        }
    }
    std::printf(fails ? "%d FAILED\n" : "i2_dp4a_test: all pass\n", fails);
    return fails ? 1 : 0;
}
