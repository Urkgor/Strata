// q36_blas_check.hpp against a CPU model of cublasGemmStridedBatchedEx, called with the arguments q36_engine.cu's prompt attention
// passes (column-major, op T / N, strideA 0, ldc = ldp or H * HD): the checks must accept the right output of every shape, and
// refuse a wrong one (one value off at the last cell / row, NaN, a transposed layout, an FP16-accumulated sum).  No GPU: it tests
// the index arithmetic of the checks, which is what would otherwise switch the engine's fast path off for nothing (or leave a wrong
// GEMM on).
#include <cmath>
#include <cstdio>
#include <vector>

#include "q36_blas_check.hpp"

// C[b] (m x n, ldc) = op(A[b]) op(B[b]), column-major, FP32 accumulation like the engine's CUBLAS_COMPUTE_32F
static void gemm_sb(bool tA, bool tB, int m, int n, int k, const float* A, int lda, long long sA, const float* B, int ldb, long long sB,
                    float* C, int ldc, long long sC, int batch, bool fp16_acc = false) {
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                float acc = 0.f;
                for (int l = 0; l < k; ++l) {
                    const float a = tA ? A[b * sA + (size_t) i * lda + l] : A[b * sA + (size_t) l * lda + i];
                    const float bb = tB ? B[b * sB + (size_t) l * ldb + j] : B[b * sB + (size_t) j * ldb + l];
                    acc += a * bb;
                    if (fp16_acc) acc = std::round(acc * 1024.f) / 1024.f;   // a crude 10-bit accumulator
                }
                C[b * sC + (size_t) j * ldc + i] = acc;
            }
}

static int fails = 0;
#define EXPECT(c, ...)                                  \
    do {                                                \
        if (!(c)) {                                     \
            ++fails;                                    \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                   \
            std::printf("\n");                          \
        }                                               \
    } while (0)

int main() {
    const int H = 16, HD = 256, HKV = 2, G = H / HKV;
    for (const q36chk::Shape& sh : q36chk::shapes()) {
        const int nt = sh.nt, P = sh.P, ldp = (P + 7) / 8 * 8;
        std::vector<float> K((size_t) P * HD), V((size_t) P * HD), Q((size_t) nt * H * HD), S((size_t) G * nt * ldp);
        q36chk::fill(K, 1);
        q36chk::fill(V, 2);
        q36chk::fill(Q, 3);
        q36chk::fill(S, 4);
        // scores: the engine's call (op T, N; m = P, n = nt, k = HD; A = K lda HD strideA 0; B = Q ldb H * HD strideB HD; C ldc ldp strideC nt * ldp)
        std::vector<float> C((size_t) G * nt * ldp, 0.f);
        gemm_sb(true, false, P, nt, HD, K.data(), HD, 0, Q.data(), H * HD, HD, C.data(), ldp, (long long) nt * ldp, G);
        double w = 0;
        EXPECT(q36chk::check_scores(C.data(), K.data(), Q.data(), H, HD, G, nt, P, ldp, &w), "scores nt %d P %d (worst %g)", nt, P, w);
        // values: (op N, N; m = HD, n = nt, k = P; A = V lda HD strideA 0; B = S ldb ldp strideB nt * ldp; C ldc H * HD strideC HD)
        std::vector<float> O((size_t) nt * H * HD, 0.f);
        gemm_sb(false, false, HD, nt, P, V.data(), HD, 0, S.data(), ldp, (long long) nt * ldp, O.data(), H * HD, HD, G);
        EXPECT(q36chk::check_values(O.data(), S.data(), V.data(), H, HD, G, nt, P, ldp, &w), "values nt %d P %d (worst %g)", nt, P, w);

        // wrong outputs are refused
        {
            std::vector<float> Cb = C;   // the last cell of the last query of the last head
            Cb[((size_t) (G - 1) * nt + (nt - 1)) * ldp + (P - 1)] += 0.25f;
            EXPECT(!q36chk::check_scores(Cb.data(), K.data(), Q.data(), H, HD, G, nt, P, ldp), "last cell wrong accepted (nt %d)", nt);
            Cb = C;
            for (float& x : Cb) x = NAN;
            EXPECT(!q36chk::check_scores(Cb.data(), K.data(), Q.data(), H, HD, G, nt, P, ldp), "NaN accepted (nt %d)", nt);
            std::vector<float> Cs((size_t) G * nt * ldp, 0.f);   // the same product with the queries and cells swapped in the layout
            gemm_sb(true, false, P, nt, HD, K.data(), HD, 0, Q.data(), H * HD, HD, Cs.data(), P, (long long) nt * P, G);
            if (ldp != P) EXPECT(!q36chk::check_scores(Cs.data(), K.data(), Q.data(), H, HD, G, nt, P, ldp), "ld mix-up accepted (nt %d)", nt);
            std::vector<float> Oc = O;
            Oc[((size_t) (nt - 1) * H + (G - 1)) * HD + (HD - 1)] -= 0.5f;
            EXPECT(!q36chk::check_values(Oc.data(), S.data(), V.data(), H, HD, G, nt, P, ldp), "last value wrong accepted (nt %d)", nt);
            std::vector<float> Oh((size_t) nt * H * HD, 0.f);   // FP16-ish accumulation
            gemm_sb(false, false, HD, nt, P, V.data(), HD, 0, S.data(), ldp, (long long) nt * ldp, Oh.data(), H * HD, HD, G, true);
            EXPECT(!q36chk::check_values(Oh.data(), S.data(), V.data(), H, HD, G, nt, P, ldp), "low-precision accumulation accepted (nt %d)", nt);
        }
    }
    (void) HKV;
    std::printf(fails ? "%d FAILED\n" : "q36_blas_check_test: all shapes pass, wrong outputs refused\n", fails);
    return fails ? 1 : 0;
}
