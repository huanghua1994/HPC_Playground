#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_proxy.h"

#define CUDA_RUNTIME_CHECK(statement)                                               \
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


int  cuda_get_device()
{
    int device;
    CUDA_RUNTIME_CHECK( cudaGetDevice(&device) );
    return device;
}

int  cuda_check_dev_p2p(const int self_dev_id, const int peer_dev_id)
{
    int can_p2p = 0;
    CUDA_RUNTIME_CHECK( cudaDeviceCanAccessPeer(&can_p2p, self_dev_id, peer_dev_id) );
    return can_p2p;
}

void cuda_get_ipc_mem_handle(void *dptr, int *handle_bytes, void **handle_)
{
    *handle_bytes = sizeof(cudaIpcMemHandle_t);
    cudaIpcMemHandle_t *handle_p = (cudaIpcMemHandle_t *) malloc(sizeof(cudaIpcMemHandle_t));
    CUDA_RUNTIME_CHECK( cudaIpcGetMemHandle(handle_p, dptr) );
    *handle_ = handle_p;
}

void cuda_open_ipc_mem_handle(void **dptr, void *handle)
{
    CUDA_RUNTIME_CHECK( cudaIpcOpenMemHandle(dptr, *((cudaIpcMemHandle_t *) handle), cudaIpcMemLazyEnablePeerAccess) );
}

void cuda_close_ipc_mem_handle(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaIpcCloseMemHandle(dptr) );
}