#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "CUDA_utils.h"
#include "test_kernel.h"
#include "msg_channel_cuda.cuh"

__global__ void test_kernel(const int64_t seed)
{
    // Let's waste some time here. Make sure seed > 0 so the 
    // compiler won't remove the loop in optimization
    int tx = threadIdx.x % 256;
    int bx = blockIdx.x  % 128;
    int fake_res = get_local_thread_id();
    for (int i = 0; i < 64 * tx + bx; i++)
        fake_res = (fake_res * 63 + bx * 31 + tx * 15) % 1023;
    double result = (seed > 0) ? (double) get_global_thread_id() : (double) fake_res;
    
    // Message is ready, issue it
    ch_issue_double_msg(result);
}

void launch_test_kernel(const int grid_dim_x, const int block_dim_x)
{
    test_kernel<<<grid_dim_x, block_dim_x>>>(1);
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
    printf("test_kernel completed\n");
}
