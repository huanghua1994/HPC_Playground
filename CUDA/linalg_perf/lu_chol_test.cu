#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_utils.cuh"
#include "cublas_cusolver_init.cuh"
#include <curand.h>

__global__ void set_diag_element_kernel(const int ncol, double *A, const int ldA, const double val)
{
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) A[i * ldA + i] = val;
    __syncthreads();
}

void test_cusolver_lu_chol(const int m, const int n_test)
{
    init_cublas_cusolver();

    int ldA = m;
    int lwork_getrf = 0, lwork_potrf = 0, lwork_potri = 0, retval;
    double *A = NULL, *A0 = NULL, *B = NULL;
    double *getrf_work = NULL, *potrf_work = NULL, *potri_work = NULL;
    int *ipiv;

    // Allocate memory on device
    CUDA_RT_CHECK( cudaMalloc((void **) &A,  sizeof(double) * m * m) );
    CUDA_RT_CHECK( cudaMalloc((void **) &A0, sizeof(double) * m * m) );
    CUDA_RT_CHECK( cudaMalloc((void **) &B,  sizeof(double) * m * m) );
    CUDA_RT_CHECK( cudaMalloc((void **) &ipiv, sizeof(int) * m) );
    CUSOLVER_CHECK( cusolverDnDgetrf_bufferSize(cusolver_dn_handle, m, m, A, ldA, &lwork_getrf) );
    CUSOLVER_CHECK( cusolverDnDpotrf_bufferSize(cusolver_dn_handle, CUBLAS_FILL_MODE_UPPER, m, A, ldA, &lwork_potrf) );
    CUSOLVER_CHECK( cusolverDnDpotri_bufferSize(cusolver_dn_handle, CUBLAS_FILL_MODE_UPPER, m, A, ldA, &lwork_potri) );
    CUDA_RT_CHECK( cudaMalloc((void **) &getrf_work, sizeof(double) * lwork_getrf) );
    CUDA_RT_CHECK( cudaMalloc((void **) &potrf_work, sizeof(double) * lwork_potrf) );
    CUDA_RT_CHECK( cudaMalloc((void **) &potri_work, sizeof(double) * lwork_potri) );
 
    // Initialize random SPD A
    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    CURAND_CHECK( curandGenerateUniformDouble(gen, A, m * m) );
    CURAND_CHECK( curandDestroyGenerator(gen) );
    CUBLAS_CHECK( cublasDgemm(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, m, 
        &d_one, A, ldA, A, ldA, &d_zero, A0, ldA
    ) );

    // Test dgeqrf and dorgqr
    cudaEvent_t *events = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * 8);
    for (int i = 0; i < 8; i++) CUDA_RT_CHECK( cudaEventCreate(&events[i]) );
    fprintf(stderr, "getrf_ms, getrs_ms, potrf_ms, potri_ms | getrf_gflops, getrs_gflops, potrf_gflops, potri_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        // LU factorization
        CUDA_RT_CHECK( cudaMemcpyAsync(A, A0, sizeof(double) * m * m, cudaMemcpyDeviceToDevice, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(events[0], cusolver_stream) );
        CUSOLVER_CHECK( cusolverDnDgetrf(
            cusolver_dn_handle, m, m, A, ldA, 
            getrf_work, ipiv, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpyAsync(&retval, cusolver_retval, sizeof(int), cudaMemcpyDeviceToHost, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(events[1], cusolver_stream) );
        CUDA_RT_CHECK( cudaEventSynchronize(events[0]) );
        CUDA_RT_CHECK( cudaEventSynchronize(events[1]) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDgetrf returned %d\n", __FILE__, __LINE__, retval);

        // Inverse matrix using LU factorization
        CUDA_RT_CHECK( cudaEventRecord(events[2], cusolver_stream) );
        CUDA_RT_CHECK( cudaMemset(B, 0, sizeof(double) * m * m) );
        set_diag_element_kernel<<<1, 1024>>>(m, B, m, 1.0);
        CUDA_RT_CHECK( cudaPeekAtLastError() );
        CUSOLVER_CHECK( cusolverDnDgetrs(
            cusolver_dn_handle, CUBLAS_OP_N, m, m, 
            A, ldA, ipiv, B, m, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpyAsync(&retval, cusolver_retval, sizeof(int), cudaMemcpyDeviceToHost, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(events[3], cusolver_stream) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDgetrs returned %d\n", __FILE__, __LINE__, retval);
        
        // Cholesky factorization
        CUDA_RT_CHECK( cudaMemcpyAsync(A, A0, sizeof(double) * m * m, cudaMemcpyDeviceToDevice, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(events[4], cusolver_stream) );
        CUSOLVER_CHECK( cusolverDnDpotrf(
            cusolver_dn_handle, CUBLAS_FILL_MODE_UPPER, m, A, ldA, 
            potrf_work, lwork_potrf, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpyAsync(&retval, cusolver_retval, sizeof(int), cudaMemcpyDeviceToHost, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(events[5], cusolver_stream) );
        CUDA_RT_CHECK( cudaEventSynchronize(events[4]) );
        CUDA_RT_CHECK( cudaEventSynchronize(events[5]) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDpotrf returned %d\n", __FILE__, __LINE__, retval);

        // Inverse matrix using Cholesky factorization
        CUDA_RT_CHECK( cudaEventRecord(events[6], cusolver_stream) );
        CUSOLVER_CHECK( cusolverDnDpotri(
            cusolver_dn_handle, CUBLAS_FILL_MODE_UPPER, m, A, ldA, 
            potri_work, lwork_potri, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpyAsync(&retval, cusolver_retval, sizeof(int), cudaMemcpyDeviceToHost, cusolver_stream) );
        CUDA_RT_CHECK( cudaEventRecord(events[7], cusolver_stream) );
        CUDA_RT_CHECK( cudaEventSynchronize(events[6]) );
        CUDA_RT_CHECK( cudaEventSynchronize(events[7]) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDpotri returned %d\n", __FILE__, __LINE__, retval);

        float getrf_ms, getrs_ms, potrf_ms, potri_ms;
        float getrf_mflops = ((2.0f/3.0f) * (float) m * (float) m * (float) m) / 1000000.0f;
        float getrs_mflops = ((2.0f/1.0f) * (float) m * (float) m * (float) m) / 1000000.0f;
        float potrf_mflops = ((1.0f/3.0f) * (float) m * (float) m * (float) m) / 1000000.0f;
        float potri_mflops = ((2.0f/3.0f) * (float) m * (float) m * (float) m) / 1000000.0f;
        CUDA_RT_CHECK( cudaEventElapsedTime(&getrf_ms, events[0], events[1]) );
        CUDA_RT_CHECK( cudaEventElapsedTime(&getrs_ms, events[2], events[3]) );
        CUDA_RT_CHECK( cudaEventElapsedTime(&potrf_ms, events[4], events[5]) );
        CUDA_RT_CHECK( cudaEventElapsedTime(&potri_ms, events[6], events[7]) );
        fprintf(
            stderr, "%.2f, %.2f, %.2f, %.2f | %.2f, %.2f, %.2f, %.2f\n", 
            getrf_ms, getrs_ms, potrf_ms, potri_ms,
            getrf_mflops / getrf_ms, getrs_mflops / getrs_ms,
            potrf_mflops / potrf_ms, potri_mflops / potri_ms
        );
    }

    CUDA_RT_CHECK( cudaFree(A) );
    CUDA_RT_CHECK( cudaFree(A0) );
    CUDA_RT_CHECK( cudaFree(B) );
    CUDA_RT_CHECK( cudaFree(ipiv) );
    CUDA_RT_CHECK( cudaFree(getrf_work) );
    CUDA_RT_CHECK( cudaFree(potrf_work) );
    CUDA_RT_CHECK( cudaFree(potri_work) );
    for (int i = 0; i < 8; i++) CUDA_RT_CHECK( cudaEventDestroy(events[i]) );
    free(events);
}

int main(int argc, char **argv)
{
    int m = 0, n_test = 10;
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s m n_test \n", argv[0]);
        return 255;
    }
    m = atoi(argv[1]);
    n_test = atoi(argv[2]);
    fprintf(stderr, "m = %d, n_test = %d\n", m, n_test);

    test_cusolver_lu_chol(m, n_test);

    return 0;
}