#ifndef __CUBLAS_CUSOLVER_INIT_CUH__
#define __CUBLAS_CUSOLVER_INIT_CUH__

#include <cublas_v2.h>
#include <cusolverDn.h>

static int    cublas_cusolve_init = 0;
static int    *cusolver_retval    = NULL;
static double d_zero = 0.0, d_one = 1.0, d_neg_one = -1.0;
static float  s_zero = 0.0, s_one = 1.0, s_neg_one = -1.0;
static cublasHandle_t     cublas_handle;
static cublasStatus_t     cublas_stat;
static cudaStream_t       cublas_stream;
static cusolverDnHandle_t cusolver_dn_handle;
static cusolverStatus_t   cusolver_stat;
static cudaStream_t       cusolver_stream;

static int init_cublas_cusolver()
{
    if (cublas_cusolve_init) return 1;
    CUBLAS_CHECK( cublasCreate(&cublas_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, cublas_stream) );
    CUSOLVER_CHECK( cusolverDnCreate(&cusolver_dn_handle) );
    // Use the same stream in cuBLAS and cuSOLVER
    CUSOLVER_CHECK( cusolverDnSetStream(cusolver_dn_handle, cublas_stream) );
    CUSOLVER_CHECK( cusolverDnGetStream(cusolver_dn_handle, &cusolver_stream) );
    CUDA_RT_CHECK( cudaMalloc((void **) &cusolver_retval, sizeof(int)) );
    cublas_cusolve_init = 1;
    return 1;
}

#endif
