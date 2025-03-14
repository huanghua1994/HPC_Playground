#pragma once

#include <cusolverDn.h>

typedef struct
{
    size_t geqrf_dev_bytes{0};
    size_t geqrf_host_bytes{0};
    int orgqr_lwork{0};

    void *geqrf_dev_buf{nullptr};
    void *geqrf_host_buf{nullptr};
    void *orgqr_work{nullptr};

    cudaEvent_t events[2];
    float geqrf_ms{0.0f};
    float orgqr_ms{0.0f};
} householder_qr_workbuf_s;

template <typename T>
void householder_qr_workbuf_alloc(
    cusolverDnHandle_t &cusolver_dn_handle, cusolverDnParams_t &cusolver_dn_params, 
    const int m, const int n, householder_qr_workbuf_s *workbuf
)
{
    int64_t m_ = static_cast<int64_t>(m);
    int64_t n_ = static_cast<int64_t>(n);
    cudaDataType dataTypeA = T_to_cuda_dtype<T>();
    void *A = nullptr;
    int64_t lda_ = m_;
    cudaDataType dataTypeTau = dataTypeA;
    void *tau = nullptr;
    cudaDataType computeType = dataTypeA;
    CUSOLVER_CHECK( cusolverDnXgeqrf_bufferSize(
        cusolver_dn_handle, cusolver_dn_params, 
        m_, n_, dataTypeA,
        A, lda_, dataTypeTau, tau, computeType,
        &workbuf->geqrf_dev_bytes, &workbuf->geqrf_host_bytes
    ) );
    CUDA_CHECK( cudaMalloc((void **) &workbuf->geqrf_dev_buf, workbuf->geqrf_dev_bytes) );
    CUDA_CHECK( cudaMallocHost((void **) &workbuf->geqrf_host_buf, workbuf->geqrf_host_bytes) );

    int lda = m;
    if constexpr(std::is_same_v<T, double>)
    {
        CUSOLVER_CHECK( cusolverDnDorgqr_bufferSize(
            cusolver_dn_handle, m, n, n,
            (double *) A, lda, (double *) tau, &workbuf->orgqr_lwork
        ) );
    }
    if constexpr(std::is_same_v<T, float>)
    {
        CUSOLVER_CHECK( cusolverDnSorgqr_bufferSize(
            cusolver_dn_handle, m, n, n,
            (float *) A, lda, (float *) tau, &workbuf->orgqr_lwork
        ) );
    }
    CUDA_CHECK( cudaMalloc((void **) &workbuf->orgqr_work, sizeof(T) * workbuf->orgqr_lwork) );

    for (int i = 0; i < 2; i++) CUDA_CHECK( cudaEventCreate(&workbuf->events[i]) );
    workbuf->geqrf_ms = 0.0f;
    workbuf->orgqr_ms = 0.0f;
}

void householder_qr_workbuf_free(householder_qr_workbuf_s *workbuf)
{
    CUDA_CHECK( cudaFree(workbuf->geqrf_dev_buf) );
    CUDA_CHECK( cudaFreeHost(workbuf->geqrf_host_buf) );
    CUDA_CHECK( cudaFree(workbuf->orgqr_work) );
    for (int i = 0; i < 2; i++) CUDA_CHECK( cudaEventDestroy(workbuf->events[i]) );
}

template <typename T>
void householder_qr(
    const int m, const int n, T *A, const int ldA, T *tau,
    const int explicit_Q, T *Q, const int ldQ,
    householder_qr_workbuf_s *workbuf, cudaStream_t stream,
    cusolverDnHandle_t cusolver_dn_handle, cusolverDnParams_t cusolver_dn_params
)
{
    float timer_tmp;
    constexpr cudaDataType dataTypeA   = T_to_cuda_dtype<T>();
    constexpr cudaDataType dataTypeTau = dataTypeA;
    constexpr cudaDataType computeType = dataTypeA;
    int *cusolver_retval_d = nullptr;
    int cusolver_retval_h;
    CUDA_CHECK( cudaMalloc((void **) &cusolver_retval_d, sizeof(int)) );
    CUSOLVER_CHECK( cusolverDnSetStream(cusolver_dn_handle, stream) );

    const int64_t ldA_ = static_cast<int64_t>(ldA);
    CUDA_CHECK( cudaEventRecord(workbuf->events[0], stream) );
    CUSOLVER_CHECK( cusolverDnXgeqrf(
        cusolver_dn_handle, cusolver_dn_params, 
        m, n, dataTypeA, A, ldA_, dataTypeTau, tau, computeType,
        workbuf->geqrf_dev_buf, workbuf->geqrf_dev_bytes,
        workbuf->geqrf_host_buf, workbuf->geqrf_host_bytes,
        cusolver_retval_d
    ) );
    CUDA_CHECK( cudaMemcpyAsync(
        &cusolver_retval_h, cusolver_retval_d, sizeof(int), 
        cudaMemcpyDeviceToHost, stream
    ) );
    CUDA_CHECK( cudaEventRecord(workbuf->events[1], stream) );
    CUDA_CHECK( cudaEventSynchronize(workbuf->events[0]) );
    CUDA_CHECK( cudaEventSynchronize(workbuf->events[1]) );
    if (cusolver_retval_h != 0)
    {
        fprintf(stderr, "cusolverDnXgeqrf failed with error code %d\n", cusolver_retval_h);
        return;
    }
    CUDA_CHECK( cudaEventElapsedTime(&timer_tmp, workbuf->events[0], workbuf->events[1]) );
    workbuf->geqrf_ms += timer_tmp;

    if (!explicit_Q)
    {
        CUDA_CHECK( cudaFree(cusolver_retval_d) );
        return;
    }

    const size_t Q_pitch = static_cast<size_t>(ldQ) * sizeof(T);
    const size_t A_pitch = static_cast<size_t>(ldA) * sizeof(T);
    const size_t width   = static_cast<size_t>(m) * sizeof(T);
    const size_t height  = static_cast<size_t>(n);
    CUDA_CHECK( cudaMemcpy2D(Q, Q_pitch, A, A_pitch, width, height, cudaMemcpyDeviceToDevice) );
    CUDA_CHECK( cudaEventRecord(workbuf->events[0], stream) );
    if constexpr(std::is_same_v<T, double>)
    {
        CUSOLVER_CHECK( cusolverDnDorgqr(
            cusolver_dn_handle, m, n, n, Q, ldQ, tau,
            (double *) workbuf->orgqr_work, workbuf->orgqr_lwork,
            cusolver_retval_d
        ) );
    }
    if constexpr(std::is_same_v<T, float>)
    {
        CUSOLVER_CHECK( cusolverDnSorgqr(
            cusolver_dn_handle, m, n, n, Q, ldQ, tau,
            (float *) workbuf->orgqr_work, workbuf->orgqr_lwork,
            cusolver_retval_d
        ) );
    }
    CUDA_CHECK( cudaEventRecord(workbuf->events[1], stream) );
    CUDA_CHECK( cudaEventSynchronize(workbuf->events[0]) );
    CUDA_CHECK( cudaEventSynchronize(workbuf->events[1]) );
    if (cusolver_retval_h != 0)
    {
        fprintf(stderr, "cusolverDnXorgqr failed with error code %d\n", cusolver_retval_h);
        return;
    }
    CUDA_CHECK( cudaEventElapsedTime(&timer_tmp, workbuf->events[0], workbuf->events[1]) );
    workbuf->orgqr_ms += timer_tmp;

    CUDA_CHECK( cudaFree(cusolver_retval_d) );
}
