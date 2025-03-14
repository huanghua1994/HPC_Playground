#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "cuda_utils.cuh"
#include "helper.cuh"
#include "block_mgs.cuh"

template <typename T>
void test_block_mgs2(const int m, const int n, T cond_num, const int blk_size, const int n_test)
{
    constexpr bool T_is_double = std::is_same_v<T, double>;
    constexpr bool T_is_float  = std::is_same_v<T, float>;
    const double d_zero = 0.0,  d_one = 1.0,  d_neg_one = -1.0;
    const float  s_zero = 0.0f, s_one = 1.0f, s_neg_one = -1.0f;
    const T T_zero = 0, T_one = 1;

    int ldA = m, ldR = n, ldE = n;
    T *A = nullptr, *A0 = nullptr, *A1 = nullptr, *R = nullptr, *E_d = nullptr, *E_h;
    CUDA_CHECK( cudaMalloc((void **) &A,   sizeof(T) * m * n) );
    CUDA_CHECK( cudaMalloc((void **) &A0,  sizeof(T) * m * n) );
    CUDA_CHECK( cudaMalloc((void **) &A1,  sizeof(T) * m * n) );
    CUDA_CHECK( cudaMalloc((void **) &R,   sizeof(T) * n * n) );
    CUDA_CHECK( cudaMalloc((void **) &E_d, sizeof(T) * n * n) );
    E_h = (T *) malloc(sizeof(T) * n * n);
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) E_h[i] = 0;
    for (int i = 0; i < n; i++) E_h[i * n + i] = 1;
    CUDA_CHECK( cudaMemcpy(E_d, E_h, sizeof(T) * n * n, cudaMemcpyHostToDevice) );

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_dn_handle;
    cusolverDnParams_t cusolver_dn_params;
    CUDA_CHECK( cudaStreamCreate(&stream) );
    CUBLAS_CHECK( cublasCreate(&cublas_handle) );
    CUSOLVER_CHECK( cusolverDnCreate(&cusolver_dn_handle) );
    CUSOLVER_CHECK( cusolverDnCreateParams(&cusolver_dn_params) );

    constexpr cudaDataType dtype_T = T_to_cuda_dtype<T>();

    block_mgs_workbuf_s *workbuf = (block_mgs_workbuf_s *) malloc(sizeof(block_mgs_workbuf_s));
    block_mgs_workbuf_alloc<T>(
        cusolver_dn_handle, cusolver_dn_params,
        m, n, blk_size, workbuf
    );
    
    rand_mat_cond_num<T>(m, n, A0, ldA, cond_num);
    printf("Random test matrix generated\n\n");

    fprintf(stderr, "row_sample, householder, trsm, chol, panel_gemm, proj_gemm  | orth_ms, buildR_ms, orth_gflops, buildR_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_CHECK( cudaMemcpy(A, A0, sizeof(T) * m * n, cudaMemcpyDeviceToDevice) );
        memset(workbuf->timers, 0, sizeof(float) * timer_idx_total);

        block_mgs<T>(
            m, n, blk_size, A, ldA, workbuf, 1,
            stream, cublas_handle, cusolver_dn_handle, cusolver_dn_params
        );
        block_mgs<T>(
            m, n, blk_size, A, ldA, workbuf, 0,
            stream, cublas_handle, cusolver_dn_handle, cusolver_dn_params
        );

        float buildR_ms;
        CUDA_CHECK( cudaEventRecord(workbuf->events[0], stream) );
        CUBLAS_CHECK( cublasGemmEx(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
            &T_one, A, dtype_T, ldA, A0, dtype_T, ldA,
            &T_zero, R, dtype_T, ldR, dtype_T, CUBLAS_GEMM_DEFAULT
        ) );
        CUDA_CHECK( cudaEventRecord(workbuf->events[1], stream) );
        CUDA_CHECK( cudaEventSynchronize(workbuf->events[0]) );
        CUDA_CHECK( cudaEventSynchronize(workbuf->events[1]) );
        CUDA_CHECK( cudaEventElapsedTime(&buildR_ms, workbuf->events[0], workbuf->events[1]) );

        float orth_mflops = 4.0f * (float) m * (float) n * (float) n / 1000000.0f;
        float buildR_mflops = 2.0f * (float) m * (float) n * (float) n / 1000000.0f;
        float orth_ms = 0.0;
        for (int i = 0; i < timer_idx_total; i++)
        {
            fprintf(stderr, "%.2f  ", workbuf->timers[i]);
            orth_ms += workbuf->timers[i];
        }
        fprintf(stderr, "| %.2f, %.2f, %.2f, %.2f\n", orth_ms, buildR_ms, orth_mflops / orth_ms, buildR_mflops / buildR_ms);
    }

    // Accuracy check
    T A0_fnorm, diff_fnorm, orth_fnorm, E_fnorm = std::sqrt(n);
    zero_lower_triangle<T>(R, ldR, n, n);
    if constexpr(T_is_double)
    {
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, 
            &d_one, A, ldA, R, ldR, &d_zero, A1, ldA
        ) );
        CUBLAS_CHECK( cublasDaxpy(cublas_handle, m * n, &d_neg_one, A0, 1, A1, 1) );
        CUBLAS_CHECK( cublasDnrm2(cublas_handle, m * n, A0, 1, &A0_fnorm) );
        CUBLAS_CHECK( cublasDnrm2(cublas_handle, m * n, A1, 1, &diff_fnorm) );
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, 
            &d_neg_one, A, ldA, A, ldA, &d_one, E_d, ldE
        ) );
        CUBLAS_CHECK( cublasDnrm2(cublas_handle, n * n, E_d, 1, &orth_fnorm) );
    }
    if constexpr(T_is_float)
    {
        CUBLAS_CHECK( cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, 
            &s_one, A, ldA, R, ldR, &s_zero, A1, ldA
        ) );
        CUBLAS_CHECK( cublasSaxpy(cublas_handle, m * n, &s_neg_one, A0, 1, A1, 1) );
        CUBLAS_CHECK( cublasSnrm2(cublas_handle, m * n, A0, 1, &A0_fnorm) );
        CUBLAS_CHECK( cublasSnrm2(cublas_handle, m * n, A1, 1, &diff_fnorm) );
        CUBLAS_CHECK( cublasSgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, 
            &s_neg_one, A, ldA, A, ldA, &s_one, E_d, ldE
        ) );
        CUBLAS_CHECK( cublasSnrm2(cublas_handle, n * n, E_d, 1, &orth_fnorm) );
    }
    fprintf(stderr, "||Q   * R - A||_{fro} / ||A||_{fro} = %.2e\n", diff_fnorm / A0_fnorm);
    fprintf(stderr, "||Q^T * Q - I||_{fro} / ||I||_{fro} = %.2e\n", orth_fnorm / E_fnorm);

    CUDA_CHECK( cudaFree(A) );
    CUDA_CHECK( cudaFree(A0) );
    CUDA_CHECK( cudaFree(A1) );
    CUDA_CHECK( cudaFree(R) );
    CUDA_CHECK( cudaFree(E_d) );
    free(E_h);

    CUDA_CHECK( cudaStreamDestroy(stream) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUSOLVER_CHECK( cusolverDnDestroy(cusolver_dn_handle) );
    CUSOLVER_CHECK( cusolverDnDestroyParams(cusolver_dn_params) );

    block_mgs_workbuf_free(workbuf);
    free(workbuf);
}

int main(int argc, char **argv)
{
    int dtype = 0, m = 0, n = 0, blk_size = 256, n_test = 10;
    double cond_num = 1e4;
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s dtype m n cond_num blk_size n_test \n", argv[0]);
        fprintf(stderr, "  - dtype    : 0 for double, 1 for float\n");
        fprintf(stderr, "  - m, n     : Test matrix is m row, n columns (m >= n)\n");
        fprintf(stderr, "  - cond_num : Condition number of the test matrix\n");
        fprintf(stderr, "  - blk_size : Block size (number of columns in a panel)\n");
        fprintf(stderr, "  - n_test   : Number of tests to run\n");
        return 255;
    }
    dtype = atoi(argv[1]);
    if (dtype < 0 || dtype > 1) dtype = 0;
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    cond_num = atof(argv[4]);
    if (argc >= 6) blk_size = atoi(argv[5]);
    if (argc >= 7) n_test = atoi(argv[6]);
    if (m < n) m = n;
    if (dtype == 0 && cond_num > 1e15) cond_num = 1e15;
    if (dtype == 1 && cond_num > 1e6) cond_num = 1e6;
    if (blk_size < 32 || blk_size > 1024) blk_size = 256;
    fprintf(
        stderr, "dtype = %d, m = %d, n = %d, cond_num = %e, blk_size = %d, n_test = %d\n", 
        dtype, m, n, cond_num, blk_size, n_test
    );

    if (dtype == 0) test_block_mgs2<double>(m, n, (double) cond_num, blk_size, n_test);
    if (dtype == 1) test_block_mgs2<float>(m, n, (float) cond_num, blk_size, n_test);

    return 0;
}