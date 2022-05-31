#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_utils.cuh"
#include "cublas_cusolver_init.cuh"
#include <curand.h>

void test_cublas_gemm(const int m, const int n, const int k, const int n_test)
{
    init_cublas_cusolver();

    int ldA = m, ldB = k, ldC = m;
    double *A = NULL, *B = NULL, *C = NULL;
    CUDA_RT_CHECK( cudaMalloc((void **) &A, sizeof(double) * m * k) );
    CUDA_RT_CHECK( cudaMalloc((void **) &B, sizeof(double) * k * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &C, sizeof(double) * m * n) );
 
    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    CURAND_CHECK( curandGenerateUniformDouble(gen, A, m * k) );
    CURAND_CHECK( curandGenerateUniformDouble(gen, B, k * n) );
    CUDA_RT_CHECK( cudaMemset(C, 0, sizeof(double) * m * n) );
    CURAND_CHECK( curandDestroyGenerator(gen) );

    cudaEvent_t gemm_start, gemm_stop;
    CUDA_RT_CHECK( cudaEventCreate(&gemm_start) );
    CUDA_RT_CHECK( cudaEventCreate(&gemm_stop) );
    fprintf(stderr, "gemm_ms, gemm_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_RT_CHECK( cudaEventRecord(gemm_start, cublas_stream) );
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
            &d_one, A, ldA, B, ldB, &d_one, C, ldC
        ) );
        CUDA_RT_CHECK( cudaEventRecord(gemm_stop, cublas_stream) );
        CUDA_RT_CHECK( cudaEventSynchronize(gemm_start) );
        CUDA_RT_CHECK( cudaEventSynchronize(gemm_stop) );

        float gemm_ms;
        float mflops = (2.0f * (float) m * (float) n * (float) k) / 1000000.0f;
        CUDA_RT_CHECK( cudaEventElapsedTime(&gemm_ms, gemm_start, gemm_stop) );
        fprintf(stderr, "%.2f, %.2f\n", gemm_ms, mflops / gemm_ms);
    }

    CUDA_RT_CHECK( cudaFree(A) );
    CUDA_RT_CHECK( cudaFree(B) );
    CUDA_RT_CHECK( cudaFree(C) );
    CUDA_RT_CHECK( cudaEventDestroy(gemm_start) );
    CUDA_RT_CHECK( cudaEventDestroy(gemm_stop) );
}

int main(int argc, char **argv)
{
    int m = 0, n = 0, k = 0, n_test = 10;
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s m n k n_test \n", argv[0]);
        return 255;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    n_test = atoi(argv[4]);
    fprintf(stderr, "m = %d, n = %d, k = %d, n_test = %d\n", m, n, k, n_test);

    test_cublas_gemm(m, n, k, n_test);

    return 0;
}