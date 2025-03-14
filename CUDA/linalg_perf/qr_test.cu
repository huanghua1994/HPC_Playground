#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "cuda_utils.cuh"
#include "helper.cuh"
#include "householder_qr.cuh"

template <typename T>
void test_householder_qr(const int m, const int n, const T cond_num, const int n_test)
{
    constexpr bool T_is_double = std::is_same_v<T, double>;
    constexpr bool T_is_float  = std::is_same_v<T, float>;
    const double d_zero = 0.0,  d_one = 1.0,  d_neg_one = -1.0;
    const float  s_zero = 0.0f, s_one = 1.0f, s_neg_one = -1.0f;

    int ldA = m, ldQ = m, ldE = n;
    T *A0 = nullptr, *A = nullptr, *A1 = nullptr, *tau = nullptr, *Q = nullptr, *E_d = NULL, *E_h;
    CUDA_CHECK( cudaMalloc((void **) &A0,  sizeof(T) * m * n) );
    CUDA_CHECK( cudaMalloc((void **) &A,   sizeof(T) * m * n) );
    CUDA_CHECK( cudaMalloc((void **) &A1,  sizeof(T) * m * n) );
    CUDA_CHECK( cudaMalloc((void **) &tau, sizeof(T) * n) );
    CUDA_CHECK( cudaMalloc((void **) &Q,   sizeof(T) * m * n) );
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

    householder_qr_workbuf_s *workbuf = (householder_qr_workbuf_s *) malloc(sizeof(householder_qr_workbuf_s));
    householder_qr_workbuf_alloc<T>(cusolver_dn_handle, cusolver_dn_params, m, n, workbuf);
    
    rand_mat_cond_num<T>(m, n, A0, ldA, cond_num);
    printf("Random test matrix generated\n\n");

    fprintf(stderr, "geqrf_ms, orgqr_ms | geqrf_gflops, orgqr_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_CHECK( cudaMemcpy(A, A0, sizeof(T) * m * n, cudaMemcpyDeviceToDevice) );
        workbuf->geqrf_ms = 0.0;
        workbuf->orgqr_ms = 0.0;

        int explicit_Q = 1;
        householder_qr<T>(
            m, n, A, ldA, tau, explicit_Q, Q, ldQ,
            workbuf, stream, cusolver_dn_handle, cusolver_dn_params
        );

        float mflops = (2.0f * (float) m * (float) n * (float) n - (2.0f/3.0f) * (float) n * (float) n * (float) n) / 1000000.0f;
        fprintf(stderr, "%.2f, %.2f | %.2f, %.2f\n", workbuf->geqrf_ms, workbuf->orgqr_ms, 
                mflops / workbuf->geqrf_ms, mflops / workbuf->orgqr_ms);
    }

    // Accuracy check
    T A0_fnorm, diff_fnorm, orth_fnorm, E_fnorm = std::sqrt(n);
    zero_lower_triangle<T>(A, ldA, m, n);
    if constexpr(T_is_double)
    {
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, 
            &d_one, Q, ldQ, A, ldA, &d_zero, A1, ldA
        ) );
        CUBLAS_CHECK( cublasDaxpy(cublas_handle, m * n, &d_neg_one, A0, 1, A1, 1) );
        CUBLAS_CHECK( cublasDnrm2(cublas_handle, m * n, A0, 1, &A0_fnorm) );
        CUBLAS_CHECK( cublasDnrm2(cublas_handle, m * n, A1, 1, &diff_fnorm) );
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, 
            &d_neg_one, Q, ldQ, Q, ldQ, &d_one, E_d, ldE
        ) );
        CUBLAS_CHECK( cublasDnrm2(cublas_handle, n * n, E_d, 1, &orth_fnorm) );
    }
    if constexpr(T_is_float)
    {
        CUBLAS_CHECK( cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, 
            &s_one, Q, ldQ, A, ldA, &s_zero, A1, ldA
        ) );
        CUBLAS_CHECK( cublasSaxpy(cublas_handle, m * n, &s_neg_one, A0, 1, A1, 1) );
        CUBLAS_CHECK( cublasSnrm2(cublas_handle, m * n, A0, 1, &A0_fnorm) );
        CUBLAS_CHECK( cublasSnrm2(cublas_handle, m * n, A1, 1, &diff_fnorm) );
        CUBLAS_CHECK( cublasSgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, 
            &s_neg_one, Q, ldQ, Q, ldQ, &s_one, E_d, ldE
        ) );
        CUBLAS_CHECK( cublasSnrm2(cublas_handle, n * n, E_d, 1, &orth_fnorm) );
    }
    fprintf(stderr, "||Q   * R - A||_{fro} / ||A||_{fro} = %.2e\n", diff_fnorm / A0_fnorm);
    fprintf(stderr, "||Q^T * Q - I||_{fro} / ||I||_{fro} = %.2e\n", orth_fnorm / E_fnorm);

    CUDA_CHECK( cudaFree(A) );
    CUDA_CHECK( cudaFree(A0) );
    CUDA_CHECK( cudaFree(A1) );
    CUDA_CHECK( cudaFree(tau) );
    CUDA_CHECK( cudaFree(Q) );
    CUDA_CHECK( cudaFree(E_d) );
    free(E_h);

    CUDA_CHECK( cudaStreamDestroy(stream) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUSOLVER_CHECK( cusolverDnDestroy(cusolver_dn_handle) );
    CUSOLVER_CHECK( cusolverDnDestroyParams(cusolver_dn_params) );

    householder_qr_workbuf_free(workbuf);
    free(workbuf);
}

int main(int argc, char **argv)
{
    int dtype = 0, m = 0, n = 0, n_test = 10;
    double cond_num = 1e4;
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s dtype m n cond_num, n_test \n", argv[0]);
        fprintf(stderr, "  - dtype  : 0 for double, 1 for float\n");
        fprintf(stderr, "  - m, n   : Test matrix is m row, n columns (m >= n)\n");
        fprintf(stderr, "  - cond_num : Condition number of the test matrix\n");
        fprintf(stderr, "  - n_test : Number of tests to run\n");
        return 255;
    }
    dtype = atoi(argv[1]);
    if (dtype < 0 || dtype > 1) dtype = 0;
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    cond_num = atof(argv[4]);
    if (argc >= 6) n_test = atoi(argv[5]);
    if (n > m) n = m;
    if (dtype == 0 && cond_num > 1e15) cond_num = 1e15;
    if (dtype == 1 && cond_num > 1e6) cond_num = 1e6;
    fprintf(stderr, "dtype = %d, m = %d, n = %d, cond_num = %e, n_test = %d\n", dtype, m, n, cond_num, n_test);

    if (dtype == 0) test_householder_qr<double>(m, n, cond_num, n_test);
    if (dtype == 1) test_householder_qr<float>(m, n, cond_num, n_test);

    return 0;
}