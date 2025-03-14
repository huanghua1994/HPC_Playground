#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "cuda_utils.cuh"
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>

template <typename T>
void test_gemm(const int m, const int n, const int k, const int n_test)
{
    const T T_zero = 0, T_one = 1;
    const int ldA = m, ldB = k, ldC = m;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    CUDA_CHECK( cudaStreamCreate(&stream) );
    CUBLAS_CHECK( cublasCreate(&cublas_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, stream) );

    T *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK( cudaMalloc((void **) &A, sizeof(T) * m * k) );
    CUDA_CHECK( cudaMalloc((void **) &B, sizeof(T) * k * n) );
    CUDA_CHECK( cudaMalloc((void **) &C, sizeof(T) * m * n) );

    constexpr cudaDataType dtype_T = T_to_cuda_dtype<T>();

    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    CURAND_CHECK( curandGenerate(gen, (unsigned int *) A, sizeof(T) * m * k / sizeof(unsigned int)) );
    CURAND_CHECK( curandGenerate(gen, (unsigned int *) B, sizeof(T) * k * n / sizeof(unsigned int)) );
    CURAND_CHECK( curandDestroyGenerator(gen) );
    CUDA_CHECK( cudaMemset(C, 0, sizeof(T) * m * n) );

    cudaEvent_t gemm_start, gemm_stop;
    CUDA_CHECK( cudaEventCreate(&gemm_start) );
    CUDA_CHECK( cudaEventCreate(&gemm_stop) );
    fprintf(stderr, "gemm_ms, gemm_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_CHECK( cudaEventRecord(gemm_start, stream) );
        CUBLAS_CHECK( cublasGemmEx(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
            &T_one, A, dtype_T, ldA, B, dtype_T, ldB,
            &T_zero, C, dtype_T, ldC, dtype_T, CUBLAS_GEMM_DEFAULT
        ) );
        CUDA_CHECK( cudaEventRecord(gemm_stop, stream) );
        CUDA_CHECK( cudaEventSynchronize(gemm_start) );
        CUDA_CHECK( cudaEventSynchronize(gemm_stop) );

        float gemm_ms;
        float mflops = (2.0f * (float) m * (float) n * (float) k) / 1000000.0f;
        CUDA_CHECK( cudaEventElapsedTime(&gemm_ms, gemm_start, gemm_stop) );
        fprintf(stderr, "%.2f, %.2f\n", gemm_ms, mflops / gemm_ms);
    }

    CUDA_CHECK( cudaEventDestroy(gemm_start) );
    CUDA_CHECK( cudaEventDestroy(gemm_stop) );

    CUDA_CHECK( cudaFree(A) );
    CUDA_CHECK( cudaFree(B) );
    CUDA_CHECK( cudaFree(C) );

    CUDA_CHECK( cudaStreamDestroy(stream) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
}

int main(int argc, char **argv)
{
    int dtype = 0, m = 0, n = 0, k = 0, n_test = 10;
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s dtype m n k n_test \n", argv[0]);
        fprintf(stderr, "  - dtype    : 0 for double, 1 for float, 2 for half\n");
        fprintf(stderr, "  - m, n, k  : Matrix A: m * k, matrix B: k * n, matrix C: m * n\n");
        fprintf(stderr, "  - n_test   : Number of tests to run\n");
        return 255;
    }
    dtype = atoi(argv[1]);
    if (dtype < 0 || dtype > 2) dtype = 0;
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    k = atoi(argv[4]);
    if (argc >= 6) n_test = atoi(argv[5]);
    fprintf(
        stderr, "dtype = %d, m = %d, n = %d, k = %d, n_test = %d\n", 
        dtype, m, n, k, n_test
    );

    if (dtype == 0) test_gemm<double>(m, n, k, n_test);
    if (dtype == 1) test_gemm<float> (m, n, k, n_test);
    if (dtype == 2) test_gemm<__half>(m, n, k, n_test);

    return 0;
}