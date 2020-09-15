#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "pthread.h"

#include "CUDA_utils.h"

__device__   uint64_t *channel_idx_d;
__device__   uint64_t *channel_buf_d;
__constant__ uint64_t channel_buf_size_d;

CUdevice  dev0;
CUcontext ctx0;
uint64_t *channel_buf_h;
uint64_t *channel_idx_h;

__global__ void test_kernel(const int64_t seed)
{
    // Let's waste some time first, make sure seed > 0 so the 
    // compiler won't remove the loop in optimization
    int tx = threadIdx.x % 256;
    int bx = blockIdx.x  % 128;
    int fake_res = get_local_thread_id();
    for (int i = 0; i < 64 * tx + bx; i++)
        fake_res = (fake_res * 63 + bx * 31 + tx * 15) % 1023;

    // Directly write the result to the channel buffer
    __threadfence();
    uint64_t global_tid = get_global_thread_id();
    uint64_t curr_idx   = atomicAdd((unsigned long long int *) channel_idx_d, 1);
    uint64_t *write_ptr = channel_buf_d + curr_idx;
    uint64_t result     = (seed > 0) ? global_tid : (uint64_t) fake_res;
    uint64_t flag       = 1;
    uint64_t payload    = ((result << 8) | flag);
    *((volatile uint64_t *) write_ptr) = payload;
}

void setup_channel(const uint64_t channel_buf_size)
{
    // Initialize CUDA device and context
    CUDA_CHECK( cuInit(0) );
    CUDA_CHECK( cuDeviceGet(&dev0, 0) );
    CUDA_CHECK( cuCtxCreate(&ctx0, dev0, 0) );

    // Allocate page-locked host memory for channel_idx_h and channel_buf_h
    CUDA_CHECK( cuMemAllocHost((void **) &channel_buf_h, sizeof(uint64_t) * channel_buf_size) );
    CUDA_CHECK( cuMemAllocHost((void **) &channel_idx_h, sizeof(uint64_t)) );
    *channel_idx_h = 0;
    memset(channel_buf_h, 0, sizeof(uint64_t) * channel_buf_size);  // Initially all payload flags are 0

    // Map page-locked channel_buf_h and channel_idx_h pointer to GPU pointer
    uint64_t *tmp_buf_dptr;
    uint64_t *tmp_idx_dptr;
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &tmp_buf_dptr, channel_buf_h, 0) );
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &tmp_idx_dptr, channel_idx_h, 0) );

    // Copy GPU pointer to GPU memory
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_buf_d, &tmp_buf_dptr, sizeof(uint64_t *), 
        0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_idx_d, &tmp_idx_dptr, sizeof(uint64_t *), 
        0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_buf_size_d, (const void*) &channel_buf_size, sizeof(uint64_t), 
        0, cudaMemcpyHostToDevice
    ));
}

void *host_daemon(void *param)
{
    uint64_t *param_  = (uint64_t *) param;
    uint64_t n_thread = param_[0];
    uint64_t result   = 0;
    uint64_t recv_idx = 0;
    uint64_t post_idx;

    while (recv_idx < n_thread)
    {
        post_idx = *((volatile uint64_t *) channel_idx_h);
        for (uint64_t i = recv_idx; i < post_idx; i++) 
        {
            uint64_t payload, flag = 0;
            while (flag != 1)
            {
                payload = *((volatile uint64_t *) (channel_buf_h + i));
                flag    = payload & (uint64_t) 0b11111111;
            }
            uint64_t addend  = payload >> 8;
            result += addend;
        }
        //if (recv_idx < post_idx) printf("[DEBUG] %d --> %d\n", recv_idx, post_idx);
        recv_idx = post_idx;
    }

    param_[1] = result;
    return NULL;
}

int main(int argc, char **argv)
{
    int grid_dim_x, block_dim_x, ret;
    printf("grid_dim_x, block_dim_x = ");
    ret = scanf("%d%d", &grid_dim_x, &block_dim_x);

    int n_thread = grid_dim_x * block_dim_x;

    const uint64_t channel_buf_size = 1024 * 1024;
    assert(n_thread <= channel_buf_size);
    setup_channel(channel_buf_size);

    uint64_t host_daemon_param[2] = {(uint64_t) n_thread, 0};
    pthread_t daemon_thread;
    ret = pthread_create(&daemon_thread, NULL, host_daemon, (void *) &host_daemon_param[0]);

    int64_t seed = (int64_t) (rand() + 1);
    test_kernel<<<grid_dim_x, block_dim_x>>>(1);

    uint64_t ref_result = 0;
    for (uint64_t i = 0; i < (uint64_t) n_thread; i++)
    {
        uint64_t addend = (i << 8);
        addend = (addend >> 8);
        ref_result += addend;
    }
    pthread_join(daemon_thread, NULL);

    if (ref_result == host_daemon_param[1]) printf("host daemon result is correct\n");
    else printf("ref_result (%llu) != host daemon result (%llu)\n", ref_result, host_daemon_param[1]);

    return 0;
}
