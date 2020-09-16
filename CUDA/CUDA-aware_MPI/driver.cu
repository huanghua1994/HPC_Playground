#include <stdio.h>
#include <assert.h>

#include "CUDA_utils.h"
#include "driver.h"

int get_gpu_device_cnt()
{
    int res;
    CUDA_RUNTIME_CHECK( cudaGetDeviceCount(&res) );
    return res;
}

void set_gpu_device(const int dev_id)
{
    CUDA_RUNTIME_CHECK( cudaSetDevice(dev_id) );
}

void memcpy_h2d(void *hptr, void *dptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr, hptr, bytes, cudaMemcpyHostToDevice) );
}

void memcpy_d2h(void *dptr, void *hptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(hptr, dptr, bytes, cudaMemcpyDeviceToHost) );
}

void alloc_gpu_mem(void **dptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMalloc(dptr_, bytes) );
}

void free_gpu_mem(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaFree(dptr) );
}

void sync_gpu_device()
{
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
}
