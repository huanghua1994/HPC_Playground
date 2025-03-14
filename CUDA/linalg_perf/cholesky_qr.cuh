#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

template <typename T>
void cholesky_qr(
    const int m, const int n, T *A, const int ldA, T *G, const int ldG,
    void *chol_dev_buf, const size_t chol_dev_bytes,
    void *chol_host_buf, const size_t chol_host_bytes,
    cudaStream_t stream, cublasHandle_t cublas_handle, int *cusolver_retval_d,
    cusolverDnHandle_t cusolver_dn_handle, cusolverDnParams_t cusolver_dn_params,
    cudaEvent_t *events, float *timer_gemm, float *timer_chol, float *timer_trsm
)
{
    constexpr cudaDataType dataTypeA   = T_to_cuda_dtype<T>();
    constexpr cudaDataType computeType = dataTypeA;
    int cusolver_retval_h;
    float gemm_ms, chol_ms, trsm_ms;

    const double d_one = 1.0;
    const float  s_one = 1.0f;
    const T T_zero = 0, T_one = 1, T_neg_one = -1;

    // (1) Compute the Gram matrix of the current panel
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[0], stream) );
    CUBLAS_CHECK( cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
        &T_one, A, dataTypeA, ldA, A, dataTypeA, ldA,
        &T_zero, G, dataTypeA, ldG, computeType, CUBLAS_GEMM_DEFAULT
    ) );
    
    // (2) Cholesky factorization of the Gram matrix
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[1], stream) );
    CUSOLVER_CHECK( cusolverDnXpotrf(
        cusolver_dn_handle, cusolver_dn_params, CUBLAS_FILL_MODE_UPPER,
        (int64_t) n, dataTypeA, G, ldG, computeType,
        chol_dev_buf, chol_dev_bytes,
        chol_host_buf, chol_host_bytes,
        cusolver_retval_d
    ) );
    CUDA_CHECK( cudaMemcpyAsync(
        &cusolver_retval_h, cusolver_retval_d, sizeof(int),
        cudaMemcpyDeviceToHost, stream
    ) );

    // (3) Apply R^-1 to the current panel
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[2], stream) );
    if constexpr(std::is_same_v<T, double>) 
    {
        CUBLAS_CHECK( cublasDtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
            &d_one, G, ldG, A, ldA
        ) );
    }
    if constexpr(std::is_same_v<T, float>) 
    {
        CUBLAS_CHECK( cublasStrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
            &s_one, G, ldG, A, ldA
        ) );
    }
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[3], stream) );

    if (events != nullptr)
    {
        CUDA_CHECK( cudaEventSynchronize(events[0]) );
        CUDA_CHECK( cudaEventSynchronize(events[1]) );
        CUDA_CHECK( cudaEventSynchronize(events[2]) );
        CUDA_CHECK( cudaEventSynchronize(events[3]) );
        CUDA_CHECK( cudaEventElapsedTime(&gemm_ms, events[0], events[1]) );
        CUDA_CHECK( cudaEventElapsedTime(&chol_ms, events[1], events[2]) );
        CUDA_CHECK( cudaEventElapsedTime(&trsm_ms, events[2], events[3]) );
        *timer_gemm += gemm_ms;
        *timer_chol += chol_ms;
        *timer_trsm += trsm_ms;
    } else {
        CUDA_CHECK( cudaStreamSynchronize(stream) );
    }
    if (cusolver_retval_h != 0)
    {
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnXpotrf returned %d\n", __FILE__, __LINE__, cusolver_retval_h);
        return;
    }
}

// Randomized XR as a preconditioner, ref: https://arxiv.org/abs/2111.11148

template <typename T>
__global__ void gather_P_rows(
    T *A, const int ldA, T *P, const int ldP,
    const int *rowidx, const int P_ncol
)
{
    T *src_ptr = A + rowidx[blockIdx.x];
    T *dst_ptr = P + blockIdx.x;
    for (int i = threadIdx.x; i < P_ncol; i += blockDim.x)
        dst_ptr[i * ldP] = src_ptr[i * ldA];
    __syncthreads();
}

template <typename T>
void randomized_xr(
    const int m, const int n, T *A, const int ldA, T *P, const int ldP,
    int *P_rowidx_h, int *P_rowidx_d, int *flags, T *tau,
    void *qr_dev_buf, const size_t qr_dev_bytes,
    void *qr_host_buf, const size_t qr_host_bytes,
    cudaStream_t stream, cublasHandle_t cublas_handle, int *cusolver_retval_d,
    cusolverDnHandle_t cusolver_dn_handle, cusolverDnParams_t cusolver_dn_params,
    cudaEvent_t *events, float *timer_row_sample, float *timer_householder, float *timer_trsm
)
{
    constexpr cudaDataType dataTypeA   = T_to_cuda_dtype<T>();
    constexpr cudaDataType dataTypeTau = dataTypeA;
    constexpr cudaDataType computeType = dataTypeA;
    int cusolver_retval_h;
    float row_sample_ms, householder_ms, trsm_ms;

    const double d_one = 1.0;
    const float  s_one = 1.0f;
    const T T_zero = 0, T_one = 1, T_neg_one = -1;

    int P_nrow = 2 * n;
    int P_ncol = n;

    // (1) Randomly select P_nrow random rows
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[0], stream) );
    memset(flags, 0, sizeof(int) * m);
    for (int i = 0; i < P_nrow; i++)
    {
        int idx = rand() % m;
        while (flags[idx]) idx = rand() % m;
        flags[idx] = 1;
        P_rowidx_h[i] = idx;
    }
    CUDA_CHECK( cudaMemcpyAsync(
        P_rowidx_d, P_rowidx_h, sizeof(int) * P_nrow,
        cudaMemcpyHostToDevice, stream
    ) );
    gather_P_rows<<<P_nrow, 32, 0, stream>>>(
        A, ldA, P, ldP, P_rowidx_d, P_ncol
    );
    CUDA_CHECK( cudaPeekAtLastError() );
    
    // (2) QR factorization of the selected rows
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[1], stream) );
    CUSOLVER_CHECK( cusolverDnXgeqrf(
        cusolver_dn_handle, cusolver_dn_params,
        P_nrow, P_ncol, dataTypeA, P, ldP,
        dataTypeTau, tau, computeType,
        qr_dev_buf, qr_dev_bytes,
        qr_host_buf, qr_host_bytes,
        cusolver_retval_d
    ) );
    CUDA_CHECK( cudaMemcpyAsync(
        &cusolver_retval_h, cusolver_retval_d, sizeof(int), 
        cudaMemcpyDeviceToHost, stream
    ) );
    
    // (3) Apply R^-1 to the current panel
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[2], stream) );
    if constexpr(std::is_same_v<T, double>)
    {
        CUBLAS_CHECK( cublasDtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
            &d_one, P, ldP, A, ldA
        ) );
    }
    if constexpr(std::is_same_v<T, float>)
    {
        CUBLAS_CHECK( cublasStrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
            &s_one, P, ldP, A, ldA
        ) );
    }
    if (events != nullptr) CUDA_CHECK( cudaEventRecord(events[3], stream) );

    if (events != nullptr)
    {
        CUDA_CHECK( cudaEventSynchronize(events[0]) );
        CUDA_CHECK( cudaEventSynchronize(events[1]) );
        CUDA_CHECK( cudaEventSynchronize(events[2]) );
        CUDA_CHECK( cudaEventSynchronize(events[3]) );
        CUDA_CHECK( cudaEventElapsedTime(&row_sample_ms,  events[0], events[1]) );
        CUDA_CHECK( cudaEventElapsedTime(&householder_ms, events[1], events[2]) );
        CUDA_CHECK( cudaEventElapsedTime(&trsm_ms,        events[2], events[3]) );
        *timer_row_sample  += row_sample_ms;
        *timer_householder += householder_ms;
        *timer_trsm        += trsm_ms;
    } else {
        CUDA_CHECK( cudaStreamSynchronize(stream) );
    }
    if (cusolver_retval_h != 0)
    {
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnXgeqrf returned %d\n", __FILE__, __LINE__, cusolver_retval_h);
        return;
    }
}
