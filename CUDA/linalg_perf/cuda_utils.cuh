#ifndef __CUDA_UTILS_CUH__
#define __CUDA_UTILS_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#define CUDA_CHECK(statement)                                                       \
    do                                                                              \
    {                                                                               \
        CUresult result = (statement);                                              \
        if (CUDA_SUCCESS != result)                                                 \
        {                                                                           \
            const char *p_err_str;                                                  \
            cuGetErrorString(result, &p_err_str);                                   \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,   \
                    p_err_str);                                                     \
            exit(-1);                                                               \
        }                                                                           \
        assert(CUDA_SUCCESS == result);                                             \
    } while (0)


#define CUDA_RT_CHECK(statement)                                                    \
    do                                                                              \
    {                                                                               \
        cudaError_t result = (statement);                                           \
        if (cudaSuccess != result)                                                  \
        {                                                                           \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,   \
                    cudaGetErrorString(result));                                    \
            exit(-1);                                                               \
        }                                                                           \
        assert(cudaSuccess == result);                                              \
    } while (0)


#define CUBLAS_CHECK(statement)                                                     \
    do                                                                              \
    {                                                                               \
        cublasStatus_t result = (statement);                                        \
        if (CUBLAS_STATUS_SUCCESS != result)                                        \
        {                                                                           \
            fprintf(stderr, "[%s:%d] cuBLAS failed\n", __FILE__, __LINE__);         \
            exit(-1);                                                               \
        }                                                                           \
        assert(CUBLAS_STATUS_SUCCESS == result);                                    \
    } while (0)
    

#define CUSOLVER_CHECK(statement)                                                   \
    do                                                                              \
    {                                                                               \
        cusolverStatus_t result = (statement);                                      \
        if (CUSOLVER_STATUS_SUCCESS != result)                                      \
        {                                                                           \
            fprintf(stderr, "[%s:%d] cuSOLVER failed\n", __FILE__, __LINE__);       \
            exit(-1);                                                               \
        }                                                                           \
        assert(CUSOLVER_STATUS_SUCCESS == result);                                  \
    } while (0)


#define CURAND_CHECK(statement)                                                     \
    do                                                                              \
    {                                                                               \
        curandStatus_t result = (statement);                                        \
        if (CURAND_STATUS_SUCCESS != result)                                        \
        {                                                                           \
            fprintf(stderr, "[%s:%d] cuRAND failed\n", __FILE__, __LINE__);         \
            exit(-1);                                                               \
        }                                                                           \
        assert(CURAND_STATUS_SUCCESS == result);                                    \
    } while (0)


#define CUDA_ERROR_STRING(result)                                                   \
    do                                                                              \
    {                                                                               \
        const char *p_err_str;                                                      \
        cuGetErrorString(result, &p_err_str);                                       \
        if (CUDA_SUCCESS != result)                                                 \
        {                                                                           \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,   \
                    p_err_str);                                                     \
        }                                                                           \
    } while (0)


#define CUDA_RUNTIME_ERROR_STRING(result)                                           \
    do                                                                              \
    {                                                                               \
        if (cudaSuccess != result)                                                  \
        {                                                                           \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,   \
                    cudaGetErrorString(result));                                    \
        }                                                                           \
    } while (0)


#define get_block_size()       (blockDim.x * blockDim.y * blockDim.z)
#define get_block_id()         ( gridDim.x *  gridDim.y *  blockIdx.z +  gridDim.x *  blockIdx.y  + blockIdx.x)
#define get_block_thread_id()  (blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x)
#define get_global_thread_id() (blockDim.x * blockDim.y *  blockDim.z * get_block_id() + get_block_thread_id())

#endif
    