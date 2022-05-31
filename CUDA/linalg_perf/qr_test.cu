#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_utils.cuh"
#include "cublas_cusolver_init.cuh"
#include <curand.h>

void test_cusolver_qr(const int m, const int n, const int n_test)
{
    init_cublas_cusolver();

    int ldA = m;
    int lwork_geqrf = 0, lwork_orgqr = 0, retval;
    double *A = NULL, *A0 = NULL, *Q = NULL, *tau = NULL;
    double *geqrf_work = NULL, *orgqr_work = NULL;

    // Allocate memory on device
    CUDA_RT_CHECK( cudaMalloc((void **) &A,   sizeof(double) * m * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &A0,  sizeof(double) * m * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &Q,   sizeof(double) * m * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &tau, sizeof(double) * n) );
    CUSOLVER_CHECK( cusolverDnDgeqrf_bufferSize(cusolver_dn_handle, m, n, A, ldA, &lwork_geqrf) );
    CUSOLVER_CHECK( cusolverDnDorgqr_bufferSize(cusolver_dn_handle, m, n, n, Q, ldA, tau, &lwork_orgqr) );
    CUDA_RT_CHECK( cudaMalloc((void **) &geqrf_work, sizeof(double) * lwork_geqrf) );
    CUDA_RT_CHECK( cudaMalloc((void **) &orgqr_work, sizeof(double) * lwork_orgqr) );
 
    // Initialize random A
    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    CURAND_CHECK( curandGenerateUniformDouble(gen, A0, m * n) );
    CURAND_CHECK( curandDestroyGenerator(gen) );

    // Test dgeqrf and dorgqr
    cudaEvent_t dgeqrf_start, dgeqrf_stop, dorgqr_start, dorgqr_stop;
    CUDA_RT_CHECK( cudaEventCreate(&dgeqrf_start) );
    CUDA_RT_CHECK( cudaEventCreate(&dgeqrf_stop) );
    CUDA_RT_CHECK( cudaEventCreate(&dorgqr_start) );
    CUDA_RT_CHECK( cudaEventCreate(&dorgqr_stop) );
    fprintf(stderr, "geqrf_ms, orgqr_ms | geqrf_gflops, orgqr_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_RT_CHECK( cudaMemcpyAsync(A, A0, sizeof(double) * m * n, cudaMemcpyDeviceToDevice, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(dgeqrf_start, cusolver_stream) );
        CUSOLVER_CHECK( cusolverDnDgeqrf(
            cusolver_dn_handle, m, n, A, ldA, tau,
            geqrf_work, lwork_geqrf, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpyAsync(&retval, cusolver_retval, sizeof(int), cudaMemcpyDeviceToHost, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(dgeqrf_stop, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventSynchronize(dgeqrf_start) );
        CUDA_RT_CHECK( cudaEventSynchronize(dgeqrf_stop) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDgeqrf returned %d\n", __FILE__, __LINE__, retval);
        
        CUDA_RT_CHECK( cudaMemcpyAsync(Q, A, sizeof(double) * m * n, cudaMemcpyDeviceToDevice, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(dorgqr_start, cusolver_stream) );
        CUSOLVER_CHECK( cusolverDnDorgqr(
            cusolver_dn_handle, m, n, n, Q, ldA, tau,
            orgqr_work, lwork_orgqr, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpyAsync(&retval, cusolver_retval, sizeof(int), cudaMemcpyDeviceToHost, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(dorgqr_stop, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventSynchronize(dorgqr_start) );
        CUDA_RT_CHECK( cudaEventSynchronize(dorgqr_stop) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDorgqr returned %d\n", __FILE__, __LINE__, retval);

        float geqrf_ms, orgqr_ms;
        float mflops = (2.0f * (float) m * (float) n * (float) n - (2.0f/3.0f) * (float) n * (float) n * (float) n) / 1000000.0f;
        CUDA_RT_CHECK( cudaEventElapsedTime(&geqrf_ms, dgeqrf_start, dgeqrf_stop) );
        CUDA_RT_CHECK( cudaEventElapsedTime(&orgqr_ms, dorgqr_start, dorgqr_stop) );
        fprintf(stderr, "%.2f, %.2f | %.2f, %.2f\n", geqrf_ms, orgqr_ms, mflops / geqrf_ms, mflops / orgqr_ms);
    }

    CUDA_RT_CHECK( cudaFree(A) );
    CUDA_RT_CHECK( cudaFree(A0) );
    CUDA_RT_CHECK( cudaFree(Q) );
    CUDA_RT_CHECK( cudaFree(tau) );
    CUDA_RT_CHECK( cudaFree(geqrf_work) );
    CUDA_RT_CHECK( cudaFree(orgqr_work) );
    CUDA_RT_CHECK( cudaEventDestroy(dgeqrf_start) );
    CUDA_RT_CHECK( cudaEventDestroy(dgeqrf_stop) );
    CUDA_RT_CHECK( cudaEventDestroy(dorgqr_start) );
    CUDA_RT_CHECK( cudaEventDestroy(dorgqr_stop) );
}

int main(int argc, char **argv)
{
    int m = 0, n = 0, n_test = 10;
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s m n n_test \n", argv[0]);
        return 255;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    n_test = atoi(argv[3]);
    fprintf(stderr, "m = %d, n = %d, n_test = %d\n", m, n, n_test);

    test_cusolver_qr(m, n, n_test);

    return 0;
}