// q36_blas_check.hpp - the check of the prompt attention's two cuBLAS GEMMs on the cache's card (--kv-gpu, q36_engine.cu).
//
// WHY.  On an NVIDIA T1000 (Turing TU117, no tensor cores) cublasGemmStridedBatchedEx answered CUBLAS_STATUS_INTERNAL_ERROR for a
// 9-query chunk (m = 2057 cells, n = 9) after a chunk of 2048 had passed.  A card and a library can also answer SUCCESS and be wrong.
// At load the engine runs the two GEMMs on a few shapes of that kind, reads the outputs back and compares samples with the host
// reference below; the prompt attention then uses cuBLAS only for query slices of at least the size from which every tested shape
// was right, and the split-K kernels (exact FP32) for the others.
//
// The data is built so that every product and every partial sum is exactly representable in FP32 (values k / 64 in half; sums of at
// most a few thousand products of multiples of 2^-12): any FP32-accumulating GEMM gives the reference to the last bit whatever its
// order, an FP16-accumulating one does not.  Host only: the engine converts the floats to halves.
//
// Layouts (the engine's): K, V [P][HD]; Q [nt][H][HD]; scores C[(g * nt + t) * ldp + p] for the G query heads g of a KV head;
// probabilities S the same; values O[(t * H + g) * HD + d].
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace q36chk {

struct Shape {
    int nt, P;   // queries in a slice, cells they see
};

/// ascending nt: the slices of a prompt chunk that is not a multiple of 8 (a tail), and the bigger ones
inline const std::vector<Shape>& shapes() {
    static const std::vector<Shape> s = {{9, 2057},  {10, 1030}, {12, 777},  {16, 1024}, {17, 513},  {24, 999},
                                         {33, 1500}, {48, 640},  {64, 2048}, {100, 700}, {128, 1031}, {256, 520}};
    return s;
}

/// k / 64 for k in [-64, 63]: exact in half, and so are the sums (see above)
inline void fill(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed * 2654435761u + 12345u;
    for (float& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (float) ((int) ((s >> 16) & 127) - 64) / 64.0f;
    }
}

struct Sample {
    int g, t, x;
};

/// the points to check: the corners and the last cells / rows of the output (where a tail goes wrong), and `count` others
inline std::vector<Sample> samples(int G, int nt, int xmax, uint32_t seed, int count) {
    std::vector<Sample> out;
    for (int g : {0, G / 2, G - 1})
        for (int t : {0, nt / 2, nt - 1})
            for (int x : {0, 1, xmax / 2, xmax - 2, xmax - 1})
                if (x >= 0 && x < xmax) out.push_back({g, t, x});
    uint32_t s = seed * 2246822519u + 7u;
    for (int i = 0; i < count; ++i) {
        s = s * 1664525u + 1013904223u;
        const int g = (int) ((s >> 8) % (uint32_t) G);
        s = s * 1664525u + 1013904223u;
        const int t = (int) ((s >> 8) % (uint32_t) nt);
        s = s * 1664525u + 1013904223u;
        out.push_back({g, t, (int) ((s >> 8) % (uint32_t) xmax)});
    }
    return out;
}

/// scores GEMM output: C[(g * nt + t) * ldp + p] = sum_d K[p * HD + d] * Q[(t * H + g) * HD + d].  `worst`: the largest error / tolerance.
inline bool check_scores(const float* C, const float* K, const float* Q, int H, int HD, int G, int nt, int P, int ldp, double* worst = nullptr) {
    bool ok = true;
    double w = 0;
    for (const Sample& s : samples(G, nt, P, (uint32_t) (nt * 31 + P), 400)) {
        double ref = 0, mag = 0;
        for (int d = 0; d < HD; ++d) {
            const double a = (double) K[(size_t) s.x * HD + d] * (double) Q[((size_t) s.t * H + s.g) * HD + d];
            ref += a;
            mag += std::fabs(a);
        }
        const double got = C[((size_t) s.g * nt + s.t) * ldp + s.x];
        const double tol = 1e-4 * mag + 1e-6;
        const double e = std::fabs(got - ref);
        if (!(e <= tol)) ok = false;   // (NaN too)
        w = std::max(w, std::isnan(e) ? 1e30 : e / tol);
    }
    if (worst) *worst = w;
    return ok;
}

/// values GEMM output: O[(t * H + g) * HD + d] = sum_p S[(g * nt + t) * ldp + p] * V[p * HD + d].
inline bool check_values(const float* O, const float* S, const float* V, int H, int HD, int G, int nt, int P, int ldp, double* worst = nullptr) {
    bool ok = true;
    double w = 0;
    for (const Sample& s : samples(G, nt, HD, (uint32_t) (nt * 17 + P), 400)) {
        double ref = 0, mag = 0;
        for (int p = 0; p < P; ++p) {
            const double a = (double) S[((size_t) s.g * nt + s.t) * ldp + p] * (double) V[(size_t) p * HD + s.x];
            ref += a;
            mag += std::fabs(a);
        }
        const double got = O[((size_t) s.t * H + s.g) * HD + s.x];
        const double tol = 1e-4 * mag + 1e-6;
        const double e = std::fabs(got - ref);
        if (!(e <= tol)) ok = false;
        w = std::max(w, std::isnan(e) ? 1e30 : e / tol);
    }
    if (worst) *worst = w;
    return ok;
}

}  // namespace q36chk
