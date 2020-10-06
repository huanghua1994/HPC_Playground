#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <unistd.h>

#include "CUDA_utils.h"
#include "test_kernel.h"

__global__ void test_kernel(const int n_msg, int *msg)
{
    printf("test_kernel launched, n_msg = %d\n", n_msg);
    int flag = 0;
    for (int i = 0; i < n_msg; i++)
    {
        int cnt = 0;
        while ( *((volatile int *) &msg[i]) != 1)
        {
            cnt++;
            if (cnt > 300000) break;
        }
        if ( *((volatile int *) &msg[i]) == 1) flag++;
    }
    printf("test_kernel returned, flag = %d\n", flag);
}

void launch_test_kernel(const int n_msg, int *msg)
{
    cudaStream_t stream;
    CUDA_RUNTIME_CHECK( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );

    for (int i = 0; i < 5; i++)
    {
        test_kernel<<<1, 1, 0, stream>>>(n_msg, msg);
        usleep(50000);
    }

    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(stream) );
    CUDA_RUNTIME_CHECK( cudaStreamDestroy(stream) );
}
