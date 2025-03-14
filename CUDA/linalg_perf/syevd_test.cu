#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_utils.cuh"
#include "cublas_cusolver_init.cuh"
#include <curand.h>

void test_cusolver_syevd(const int m, const int n_test)
{
    init_cublas_cusolver();

    int ldA = m;
    int lwork_syevd = 0, retval;
    double *A = NULL, *W = NULL, *A0 = NULL, *syevd_work = NULL;

    // Allocate memory on device
    CUDA_CHECK( cudaMalloc((void **) &A,  sizeof(double) * m * m) );
    CUDA_CHECK( cudaMalloc((void **) &A0, sizeof(double) * m * m) );
    CUDA_CHECK( cudaMalloc((void **) &W,  sizeof(double) * m) );
    CUSOLVER_CHECK( cusolverDnDsyevd_bufferSize(
        cusolver_dn_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 
        m, A, ldA, W, &lwork_syevd
    ) );
    CUDA_CHECK( cudaMalloc((void **) &syevd_work, sizeof(double) * lwork_syevd) );

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
    cudaEvent_t syevd_start, syevd_stop;
    CUDA_CHECK( cudaEventCreate(&syevd_start) );
    CUDA_CHECK( cudaEventCreate(&syevd_stop) );
    fprintf(stderr, "syevd_ms\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_CHECK( cudaMemcpyAsync(A, A0, sizeof(double) * m * m, cudaMemcpyDeviceToDevice, cusolver_stream) );
        CUDA_CHECK( cudaEventRecord(syevd_start, cusolver_stream) );
        CUSOLVER_CHECK( cusolverDnDsyevd(
            cusolver_dn_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 
            m, A, ldA, W, syevd_work, lwork_syevd, cusolver_retval
        ) );
        CUDA_CHECK( cudaMemcpyAsync(&retval, cusolver_retval, sizeof(int), cudaMemcpyDeviceToHost, cusolver_stream) );
        CUDA_CHECK( cudaEventRecord(syevd_stop, cusolver_stream) );
        CUDA_CHECK( cudaEventSynchronize(syevd_start) );
        CUDA_CHECK( cudaEventSynchronize(syevd_stop) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDsyevd returned %d\n", __FILE__, __LINE__, retval);

        float syevd_ms;
        CUDA_CHECK( cudaEventElapsedTime(&syevd_ms, syevd_start, syevd_stop) );
        fprintf(stderr, "%.2f\n", syevd_ms);
    }

    CUDA_CHECK( cudaFree(A) );
    CUDA_CHECK( cudaFree(A0) );
    CUDA_CHECK( cudaFree(W) );
    CUDA_CHECK( cudaFree(syevd_work) );
    CUDA_CHECK( cudaEventDestroy(syevd_start) );
    CUDA_CHECK( cudaEventDestroy(syevd_stop) );
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

    test_cusolver_syevd(m, n_test);

    return 0;
}