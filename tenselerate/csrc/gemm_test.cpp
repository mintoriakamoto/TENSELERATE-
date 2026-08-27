// Host test for the int8 GEMM reference. Runs on any CI with no GPU: it pins the
// integer-accumulate + per-row rescale semantics that the CUDA kernel must match.
// The CUDA kernel is validated against this same reference on a GPU runner.
#include "int8_gemm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using tenselerate::int8_gemm_ref;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
    else       std::printf("ok  : %s\n", what);
}

int main() {
    // 1. exactness: scales = 1 -> C is the exact integer dot product
    {
        const int M = 2, N = 3, K = 4;
        std::vector<int8_t> A = {1, 2, 3, 4,   -1, -2, -3, -4};
        std::vector<int8_t> B = {1, 0, 0, 0,   0, 1, 0, 0,   1, 1, 1, 1};
        std::vector<float> as = {1.f, 1.f}, bs = {1.f, 1.f, 1.f};
        std::vector<float> C(M * N, 0.f);
        int8_gemm_ref(A.data(), as.data(), B.data(), bs.data(), C.data(), M, N, K);
        // row0 dot col0 = 1; dot col1 = 2; dot col2 = 1+2+3+4 = 10
        check(C[0] == 1.f && C[1] == 2.f && C[2] == 10.f, "row0 integer dot");
        // row1 is the negation
        check(C[3] == -1.f && C[4] == -2.f && C[5] == -10.f, "row1 integer dot");
    }

    // 2. per-row scales apply on the m and n axes independently
    {
        const int M = 1, N = 2, K = 2;
        std::vector<int8_t> A = {2, 2};
        std::vector<int8_t> B = {3, 0,   0, 3};
        std::vector<float> as = {0.5f}, bs = {2.0f, 10.0f};
        std::vector<float> C(M * N, 0.f);
        int8_gemm_ref(A.data(), as.data(), B.data(), bs.data(), C.data(), M, N, K);
        // acc0 = 2*3 = 6 -> 6 * 0.5 * 2 = 6 ; acc1 = 2*3 = 6 -> 6*0.5*10 = 30
        check(std::fabs(C[0] - 6.0f) < 1e-6, "scale on n=0");
        check(std::fabs(C[1] - 30.0f) < 1e-6, "scale on n=1");
    }

    // 3. saturating extremes do not overflow int32 for reasonable K
    {
        const int M = 1, N = 1, K = 1024;
        std::vector<int8_t> A(K, 127), B(K, 127);
        std::vector<float> as = {1.f}, bs = {1.f};
        std::vector<float> C(1, 0.f);
        int8_gemm_ref(A.data(), as.data(), B.data(), bs.data(), C.data(), M, N, K);
        check(C[0] == (float)(127 * 127 * K), "K=1024 no overflow");
    }

    if (failures == 0) { std::printf("all gemm reference checks passed\n"); return 0; }
    std::printf("%d failure(s)\n", failures);
    return 1;
}
