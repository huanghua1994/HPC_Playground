#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "cuda_utils.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>

void test_cublaslt_gemm(
    const int dtype, const int m, const int n, const int k, 
    cublasOperation_t transa, cublasOperation_t transb, const int n_test
)
{
    cudaStream_t stream;
    cublasLtHandle_t lt_handle;
    CUDA_CHECK( cudaStreamCreate(&stream) );
    CUBLAS_CHECK( cublasLtCreate(&lt_handle) );

    cublasLtMatmulDesc_t operation_desc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int retrun_result = 0;
    cublasLtMatmulHeuristicResult_t heur_result = {};

    const double f64_one = 1, f64_zero = 0;
    const float  f32_one = 1, f32_zero = 0;
    const __half f16_one = 1, f16_zero = 0;
    const int lda = transa ? k : m;
    const int ldb = transb ? n : k;
    const int ldc = m;

    cublasComputeType_t compute_type;
    cudaDataType_t AB_type, C_type, scale_type;
    size_t AB_elem_bytes, C_elem_bytes;
    void *alpha = nullptr, *beta = nullptr;
    // FP64, FP32, FP16: see https://docs.nvidia.com/cuda/cublas/#id81
    if (dtype == 0)
    {
        compute_type = CUBLAS_COMPUTE_64F;
        AB_type = CUDA_R_64F;
        C_type = CUDA_R_64F;
        scale_type = CUDA_R_64F;
        AB_elem_bytes = 8;
        C_elem_bytes = 8;
        alpha = (void *) &f64_one;
        beta = (void *) &f64_zero;
    }
    if (dtype == 1)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        AB_type = CUDA_R_32F;
        C_type = CUDA_R_32F;
        scale_type = CUDA_R_32F;
        AB_elem_bytes = 4;
        C_elem_bytes = 4;
        alpha = (void *) &f32_one;
        beta = (void *) &f32_zero;
    }
    if (dtype == 2)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        AB_type = CUDA_R_16F;
        C_type = CUDA_R_16F;
        scale_type = CUDA_R_32F;
        AB_elem_bytes = 2;
        C_elem_bytes = 2;
        alpha = (void *) &f32_one;
        beta = (void *) &f32_zero;
    }
    if (dtype == 3)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        AB_type = CUDA_R_16BF;
        C_type = CUDA_R_16BF;
        scale_type = CUDA_R_32F;
        AB_elem_bytes = 2;
        C_elem_bytes = 2;
        alpha = (void *) &f32_one;
        beta = (void *) &f32_zero;
    }
    // FP8: see https://docs.nvidia.com/cuda/cublas/#id83
    int8_t fast_acc_mode = 0;
    if (dtype == 4 || dtype == 5)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        AB_type = CUDA_R_8F_E4M3;
        C_type = CUDA_R_16F;
        scale_type = CUDA_R_32F;
        AB_elem_bytes = 1;
        C_elem_bytes = 2;
        alpha = (void *) &f32_one;
        beta = (void *) &f32_zero;
        if (dtype == 4) fast_acc_mode = 1;
        if (dtype == 5) fast_acc_mode = 0;
    }

    CUBLAS_CHECK( cublasLtMatmulDescCreate(&operation_desc, compute_type, scale_type) );
    CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) );
    CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) );
    if (dtype == 4 || dtype == 5)
    {
        CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_acc_mode, sizeof(fast_acc_mode)) );
    }

    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Adesc, AB_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda) );
    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Bdesc, AB_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb) );
    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Cdesc, C_type, m, n, ldc));

    const size_t workspace_bytes = 32 * 1024 * 1024;
    CUBLAS_CHECK( cublasLtMatmulPreferenceCreate(&preference) );
    CUBLAS_CHECK( cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)) );
    CUBLAS_CHECK( cublasLtMatmulAlgoGetHeuristic(lt_handle, operation_desc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heur_result, &retrun_result));
    if (retrun_result == 0) CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);

    void *A = nullptr, *B = nullptr, *C = nullptr, *workspace = nullptr;
    CUDA_CHECK( cudaMalloc((void **) &A, AB_elem_bytes * m * k) );
    CUDA_CHECK( cudaMalloc((void **) &B, AB_elem_bytes * k * n) );
    CUDA_CHECK( cudaMalloc((void **) &C, C_elem_bytes * m * n) );
    CUDA_CHECK( cudaMalloc((void **) &workspace, workspace_bytes) );

    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    CURAND_CHECK( curandGenerate(gen, (unsigned int *) A, AB_elem_bytes * m * k / sizeof(unsigned int)) );
    CURAND_CHECK( curandGenerate(gen, (unsigned int *) B, AB_elem_bytes * k * n / sizeof(unsigned int)) );
    CURAND_CHECK( curandDestroyGenerator(gen) );
    CUDA_CHECK( cudaMemset(C, 0, C_elem_bytes * m * n) );

    cudaEvent_t gemm_start, gemm_stop;
    CUDA_CHECK( cudaEventCreate(&gemm_start) );
    CUDA_CHECK( cudaEventCreate(&gemm_stop) );

    const int n_warmup = 3;
    fprintf(stderr, "gemm_ms, gemm_gflops\n");
    for (int i_test = 0; i_test < n_warmup + n_test; i_test++)
    {
        CUDA_CHECK( cudaEventRecord(gemm_start, stream) );
        CUBLAS_CHECK( cublasLtMatmul(
            lt_handle, operation_desc, alpha, A, Adesc, B, Bdesc,
            beta, C, Cdesc, C, Cdesc, &heur_result.algo, 
            workspace, workspace_bytes, stream
        ) );
        CUDA_CHECK( cudaEventRecord(gemm_stop, stream) );
        CUDA_CHECK( cudaEventSynchronize(gemm_start) );
        CUDA_CHECK( cudaEventSynchronize(gemm_stop) );

        float gemm_ms;
        float mflops = (2.0f * (float) m * (float) n * (float) k) / 1000000.0f;
        CUDA_CHECK( cudaEventElapsedTime(&gemm_ms, gemm_start, gemm_stop) );
        if (i_test >= n_warmup) fprintf(stderr, "%.2f, %.2f\n", gemm_ms, mflops / gemm_ms);
    }

    CUDA_CHECK( cudaEventDestroy(gemm_start) );
    CUDA_CHECK( cudaEventDestroy(gemm_stop) );

    CUDA_CHECK( cudaFree(A) );
    CUDA_CHECK( cudaFree(B) );
    CUDA_CHECK( cudaFree(C) );
    CUDA_CHECK( cudaFree(workspace) );

    CUDA_CHECK( cudaStreamDestroy(stream) );
    CUBLAS_CHECK( cublasLtDestroy(lt_handle) );
}

int main(int argc, char **argv)
{
    int dtype = 0, m = 0, n = 0, k = 0, transa = 0, transb = 0, n_test = 10;
    if (argc < 7)
    {
        fprintf(stderr, "Usage: %s dtype m n k transa transb n_test \n", argv[0]);
        fprintf(stderr, "  - dtype      : 0 : fp64\n");
        fprintf(stderr, "  -            : 1 : fp32\n");
        fprintf(stderr, "  -            : 2 : fp16\n");
        fprintf(stderr, "  -            : 3 : bf16\n");
        fprintf(stderr, "  -            : 4 : f8e4m3 with fast accumulation\n");
        fprintf(stderr, "  -            : 5 : f8e4m3 without fast accumulation\n");
        fprintf(stderr, "  - m, n, k    : op(A): m * k, op(B): k * n, matrix C: m * n\n");
        fprintf(stderr, "  - trans{a,b} : 0 for no transpose, 1 for transpose\n");
        fprintf(stderr, "  - n_test     : Number of tests to run\n");
        return 255;
    }
    dtype = atoi(argv[1]);
    if (dtype < 0 || dtype > 5) dtype = 0;
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    k = atoi(argv[4]);
    transa = atoi(argv[5]);
    transb = atoi(argv[6]);
    if (argc >= 8) n_test = atoi(argv[7]);
    fprintf(
        stderr, "dtype = %d, m = %d, n = %d, k = %d, transa = %d, transb = %d, n_test = %d\n", 
        dtype, m, n, k, transa, transb, n_test
    );

    auto transa_ = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transb_ = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    test_cublaslt_gemm(dtype, m, n, k, transa_, transb_, n_test);

    return 0;
}