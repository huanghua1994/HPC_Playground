#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cufft.h>

// ========== For normal CUDA calls ========== 

static void cudaCheckCore(cudaError_t code, const char *file, int line) 
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA Error %d : %s, at %s:%d\n", code, cudaGetErrorString(code), file, line);
        exit(code);
    }
}
 
#define cudaCheck(test)      { cudaCheckCore((test), __FILE__, __LINE__); }
#define cudaCheckAfterCall() { cudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }

// ========== For cuBLAS library calls ========== 

static const char *cuBLASGetErrorString(cublasStatus_t error) 
{
    switch (error) 
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

static void cuBLASCheckCore(cublasStatus_t code, const char *file, int line) 
{
    if (code != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf(stderr,"cuBLAS Error %d : %s, at %s:%d\n", code, cuBLASGetErrorString(code), file, line);
        exit(code);
    }
}
 
#define cuBLASCheck(test) { cuBLASCheckCore((test), __FILE__, __LINE__); }

// ========== For cuSPARSE library calls ========== 

static const char *cuSPARSEGetErrorString(cusparseStatus_t error)
{
    // From: http://berenger.eu/blog/cusparse-cccuda-sparse-matrix-examples-csr-bcsr-spmv-and-conversions/
    // Read more at: http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";
            
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";
            
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";
            
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
            
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";
     
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";
     
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";
     
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";
     
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
 
    return "<unknown>";
}

static void cuSPARSECheckCore(cusparseStatus_t code, const char *file, int line) 
{
    if (code != CUSPARSE_STATUS_SUCCESS) 
    {
        fprintf(stderr,"cuSPARSE Error %d : %s, at %s:%d\n", code, cuSPARSEGetErrorString(code), file, line);
        exit(code);
    }
}
 
#define cuSPARSECheck(test) { cuSPARSECheckCore((test), __FILE__, __LINE__); }

// ========== For cuFFT library calls ========== 

static const char *cuFFTGetErrorString(cufftResult error) 
{
    switch (error) 
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}

static void cuFFTCheckCore(cufftResult code, const char *file, int line) 
{
    if (code != CUFFT_SUCCESS) 
    {
        fprintf(stderr,"cuFFT Error %d : %s, at %s:%d\n", code, cuFFTGetErrorString(code), file, line);
        exit(code);
    }
}
 
#define cuFFTCheck(test) { cuFFTCheckCore((test), __FILE__, __LINE__); }

#endif
 