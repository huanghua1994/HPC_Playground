#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cholesky_qr.cuh"

typedef enum
{
    timer_idx_row_sample = 0,
    timer_idx_householder,
    timer_idx_trsm,
    timer_idx_chol,
    timer_idx_panel_gemm,
    timer_idx_proj_gemm,
    timer_idx_total
} block_mgs_timer_idx_t;

typedef struct
{
    size_t qr_dev_bytes{0};
    size_t qr_host_bytes{0};
    size_t chol_dev_bytes{0};
    size_t chol_host_bytes{0};
    size_t misc_dev_bytes{0};
    size_t misc_host_bytes{0};

    void *qr_dev_buf{nullptr};
    void *qr_host_buf{nullptr};
    void *chol_dev_buf{nullptr};
    void *chol_host_buf{nullptr};
    void *misc_dev_buf{nullptr};
    void *misc_host_buf{nullptr};

    cudaEvent_t events[4];
    float timers[timer_idx_total];
} block_mgs_workbuf_s;

template <typename T>
void block_mgs_workbuf_alloc(
    cusolverDnHandle_t &cusolver_dn_handle, cusolverDnParams_t &cusolver_dn_params, 
    const int m, const int n, const int blk_size, block_mgs_workbuf_s *workbuf
)
{
    int64_t qr_m = static_cast<int64_t>(blk_size) * 2;
    int64_t qr_n = static_cast<int64_t>(blk_size);
    cudaDataType dataTypeA = T_to_cuda_dtype<T>();
    void *A = nullptr;
    int64_t lda = qr_m;
    cudaDataType dataTypeTau = dataTypeA;
    void *tau = nullptr;
    cudaDataType computeType = dataTypeA;
    CUSOLVER_CHECK( cusolverDnXgeqrf_bufferSize(
        cusolver_dn_handle, cusolver_dn_params, 
        qr_m, qr_n, dataTypeA,
        A, lda, dataTypeTau, tau, computeType,
        &workbuf->qr_dev_bytes, &workbuf->qr_host_bytes
    ) );
    CUDA_CHECK( cudaMalloc((void **) &workbuf->qr_dev_buf, workbuf->qr_dev_bytes) );
    CUDA_CHECK( cudaMallocHost((void **) &workbuf->qr_host_buf, workbuf->qr_host_bytes) );

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    int64_t chol_n = blk_size;
    lda = n;
    CUSOLVER_CHECK( cusolverDnXpotrf_bufferSize(
        cusolver_dn_handle, cusolver_dn_params, 
        uplo, chol_n, dataTypeA, A, lda, computeType,
        &workbuf->chol_dev_bytes, &workbuf->chol_host_bytes
    ) );
    CUDA_CHECK( cudaMalloc((void **) &workbuf->chol_dev_buf, workbuf->chol_dev_bytes) );
    CUDA_CHECK( cudaMallocHost((void **) &workbuf->chol_host_buf, workbuf->chol_host_bytes) );

    size_t P_nrow   = blk_size * 2;
    size_t P_size   = P_nrow * blk_size;    // Current panel row samples
    size_t tau_size = blk_size;             // Housesholder QR tau for P
    size_t G_size   = blk_size * blk_size;  // CholQR Gram matrix
    size_t H_size   = blk_size * n;         // Projection of current panel to remaining columns
    workbuf->misc_dev_bytes  = sizeof(T) * (P_nrow + P_size + tau_size + G_size + H_size);
    workbuf->misc_host_bytes = sizeof(int) * (P_nrow + blk_size);
    CUDA_CHECK( cudaMalloc((void **) &workbuf->misc_dev_buf, workbuf->misc_dev_bytes) );
    CUDA_CHECK( cudaMallocHost((void **) &workbuf->misc_host_buf, workbuf->misc_host_bytes) );

    for (int i = 0; i < 4; i++) CUDA_CHECK( cudaEventCreate(&workbuf->events[i]) );
    memset(workbuf->timers, 0, sizeof(float) * timer_idx_total);
}

void block_mgs_workbuf_free(block_mgs_workbuf_s *workbuf)
{
    CUDA_CHECK( cudaFree(workbuf->qr_dev_buf) );
    CUDA_CHECK( cudaFree(workbuf->chol_dev_buf) );
    CUDA_CHECK( cudaFree(workbuf->misc_dev_buf) );
    CUDA_CHECK( cudaFreeHost(workbuf->qr_host_buf) );
    CUDA_CHECK( cudaFreeHost(workbuf->chol_host_buf) );
    CUDA_CHECK( cudaFreeHost(workbuf->misc_host_buf) );
    for (int i = 0; i < 4; i++) CUDA_CHECK( cudaEventDestroy(workbuf->events[i]) );
}

template <typename T>
void block_mgs(
    const int m, const int n, const int blk_size, T *A,
    const int ldA, block_mgs_workbuf_s *workbuf, const int precond_panel,
    cudaStream_t stream, cublasHandle_t cublas_handle, 
    cusolverDnHandle_t cusolver_dn_handle, cusolverDnParams_t cusolver_dn_params
)
{
    const size_t P_nrow   = blk_size * 2;
    const size_t P_size   = P_nrow * blk_size;    // Current panel row samples
    const size_t tau_size = blk_size;             // Housesholder QR tau for P
    const size_t G_size   = blk_size * blk_size;  // CholQR Gram matrix
    int *P_rowidx_d = (int *) workbuf->misc_dev_buf;
    int *P_rowidx_h = (int *) workbuf->misc_host_buf;
    int *flags      = P_rowidx_h + P_nrow;
    T *misc_dev_buf_ = ((T *) workbuf->misc_dev_buf) + P_nrow;
    T *P   = misc_dev_buf_;
    T *tau = P + P_size;
    T *G   = tau + tau_size;
    T *H   = G + G_size;

    constexpr cudaDataType dataTypeA   = T_to_cuda_dtype<T>();
    constexpr cudaDataType computeType = dataTypeA;
    int *cusolver_retval_d = nullptr;
    CUDA_CHECK( cudaMalloc((void **) &cusolver_retval_d, sizeof(int)) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, stream) );
    CUSOLVER_CHECK( cusolverDnSetStream(cusolver_dn_handle, stream) );

    const T T_zero = 0, T_one = 1, T_neg_one = -1;

    for (int s_col = 0; s_col < n; s_col += blk_size)
    {
        int curr_bs = blk_size < (n - s_col) ? blk_size : (n - s_col);
        T *A_scol = A + s_col * ldA;

        // 1. Randomized XR as a preconditioner, https://arxiv.org/abs/2111.11148
        if (precond_panel)
        {
            int curr_P_nrow = 2 * curr_bs;
            randomized_xr<T>(
                m, curr_bs, A_scol, ldA, P, curr_P_nrow,
                P_rowidx_h, P_rowidx_d, flags, tau,
                workbuf->qr_dev_buf, workbuf->qr_dev_bytes,
                workbuf->qr_host_buf, workbuf->qr_host_bytes,
                stream, cublas_handle, cusolver_retval_d,
                cusolver_dn_handle, cusolver_dn_params,
                workbuf->events, &workbuf->timers[timer_idx_row_sample],
                &workbuf->timers[timer_idx_householder], &workbuf->timers[timer_idx_trsm]
            );
        }

        // 2. CholQR factorization of the current panel
        cholesky_qr<T>(
            m, curr_bs, A_scol, ldA, G, curr_bs,
            workbuf->chol_dev_buf, workbuf->chol_dev_bytes,
            workbuf->chol_host_buf, workbuf->chol_host_bytes,
            stream, cublas_handle, cusolver_retval_d,
            cusolver_dn_handle, cusolver_dn_params,
            &workbuf->events[0], &workbuf->timers[timer_idx_panel_gemm],
            &workbuf->timers[timer_idx_chol], &workbuf->timers[timer_idx_trsm]
        );

        // 3. Subtract current panel from the vectors behind current panel 
        if (s_col + curr_bs == n) continue;
        int ldH = blk_size;
        int n_rem_col = n - (s_col + curr_bs);
        T *A_ecol = A + (s_col + curr_bs) * ldA;
        CUDA_CHECK( cudaEventRecord(workbuf->events[0], stream) );
        CUBLAS_CHECK( cublasGemmEx(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, curr_bs, n_rem_col, m,
            &T_one, A_scol, dataTypeA, ldA, A_ecol, dataTypeA, ldA,
            &T_zero, H, dataTypeA, ldH, computeType, CUBLAS_GEMM_DEFAULT
        ) );
        CUBLAS_CHECK( cublasGemmEx(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n_rem_col, curr_bs,
            &T_neg_one, A_scol, dataTypeA, ldA, H, dataTypeA, ldH,
            &T_one, A_ecol, dataTypeA, ldA, computeType, CUBLAS_GEMM_DEFAULT
        ) );
        CUDA_CHECK( cudaEventRecord(workbuf->events[1], stream) );
        float gemm_ms;
        for (int i = 0; i < 2; i++) CUDA_CHECK( cudaEventSynchronize(workbuf->events[i]) );
        CUDA_CHECK( cudaEventElapsedTime(&gemm_ms, workbuf->events[0], workbuf->events[1]) );
        workbuf->timers[timer_idx_proj_gemm] += gemm_ms;
    }

    CUDA_CHECK( cudaFree(cusolver_retval_d) );
}
