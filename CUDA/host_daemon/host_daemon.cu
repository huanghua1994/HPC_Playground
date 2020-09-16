#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "pthread.h"

#include "CUDA_utils.h"

#define MSG_PAYLOAD_LEN                 2
#define CHANNEL_PAYLOAD_BUF_SIZE_LOG2   16
#define CHANNEL_PAYLOAD_BUF_SIZE        (1 << CHANNEL_PAYLOAD_BUF_SIZE_LOG2)
#define LO32B_MASK                      0x00000000ffffffff
#define HI32B_MASK                      0xffffffff00000000

typedef struct msg_payload
{
    uint64_t data[MSG_PAYLOAD_LEN];
} msg_payload_s, *msg_payload_p;

// high ------------------> low
//         32 |       24 |    8
//  value_low | reserved | flag 
typedef struct msg_payload_unit_0
{
    volatile uint8_t flag;
    uint8_t  resv0;
    uint16_t resv1;
    uint32_t value_low;
} msg_payload_unit_0_s, *msg_payload_unit_0_p;

// high ------------------> low
//        32  |       24 |    8
// value_high | reserved | flag 
typedef struct msg_payload_unit_1
{
    volatile uint8_t flag;
    uint8_t  resv0;
    uint16_t resv1;
    uint32_t value_high;
} msg_payload_unit_1_s, *msg_payload_unit_1_p;

// {en,de}queue_idx are always increasing. Since it is 64 bit integer, 
// practically it will not overflow. 
__device__   uint64_t *channel_payload_buf_d;           // Allocated on host, GPU writes, CPU reads
__device__   uint64_t *channel_enqueue_idx_d;           // Allocated on host, GPU writes, CPU reads
__device__   uint64_t *channel_dequeue_idx_d;           // Allocated on host, CPU writes, GPU reads
__device__   uint64_t channel_local_dequeue_idx_d;      // Only shared between GPU threads, to reduce access to *channel_dequeue_idx_d
__constant__ uint64_t channel_payload_buf_size_d;
__constant__ uint64_t channel_payload_buf_size_log2_d;

CUdevice  dev0;
CUcontext ctx0;
uint64_t *channel_payload_buf_h;
uint64_t *channel_enqueue_idx_h;
uint64_t *channel_dequeue_idx_h;
uint64_t channel_local_dequeue_idx_h;
uint64_t channel_payload_buf_size_h;
uint64_t channel_payload_buf_size_log2_h;

__device__ inline void wait_until_added_ge_val(volatile uint64_t *addr, const uint64_t addend, const uint64_t val)
{
    uint64_t val_at_addr = *addr;
    while (val_at_addr + addend < val) val_at_addr = *addr;
}

// A cyclic queue flag is either 0 or 1. Let queue_size = (1 << queue_size_log2).
// If 0 <= idx <= queue_size-1, it will be 1. Then for the next queue_size idx 
// values it will be 0, and so on.
__host__ __device__ inline uint64_t get_cyclic_queue_flag(const uint64_t queue_size_log2, const uint64_t idx)
{
    return !((idx >> queue_size_log2) & 1);
}

__device__ inline void check_queue_availability(const uint64_t enqueue_idx)
{
    uint64_t dequeue_idx = *((volatile uint64_t *) &channel_local_dequeue_idx_d);
    if (dequeue_idx + channel_payload_buf_size_d - 1 < enqueue_idx)
    {
        wait_until_added_ge_val(channel_dequeue_idx_d, channel_payload_buf_size_d - 1, enqueue_idx);
        dequeue_idx = *((volatile uint64_t *) channel_dequeue_idx_d);
        atomicMax((unsigned long long int *) &channel_local_dequeue_idx_d, dequeue_idx);
        __threadfence_system();
    }
}

__global__ void test_kernel(const int64_t seed)
{
    // Let's waste some time first, make sure seed > 0 so the 
    // compiler won't remove the loop in optimization
    int tx = threadIdx.x % 256;
    int bx = blockIdx.x  % 128;
    int fake_res = get_local_thread_id();
    for (int i = 0; i < 64 * tx + bx; i++)
        fake_res = (fake_res * 63 + bx * 31 + tx * 15) % 1023;
    double   result     = (seed > 0) ? (double) get_global_thread_id() : (double) fake_res;
    uint64_t result_buf = *((uint64_t *) &result);

    // Get channel message buffer enqueue index and wait until it is safe to enqueue
    __threadfence();
    uint64_t enqueue_idx = atomicAdd((unsigned long long int *) channel_enqueue_idx_d, 1);
    check_queue_availability(enqueue_idx);

    // Assemble message payload
    uint64_t enqueue_flag   = get_cyclic_queue_flag(channel_payload_buf_size_log2_d, enqueue_idx);
    uint64_t payload_unit_0 = ((result_buf << 32)        | enqueue_flag);
    uint64_t payload_unit_1 = ((result_buf & HI32B_MASK) | enqueue_flag);

    // Write message payload to channel message buffer
    uint64_t write_idx  = enqueue_idx & (channel_payload_buf_size_d - 1);
    uint64_t *write_ptr = channel_payload_buf_d + write_idx * MSG_PAYLOAD_LEN;
    *((volatile uint64_t *) write_ptr) = payload_unit_0;
    write_ptr++;
    *((volatile uint64_t *) write_ptr) = payload_unit_1;
}

void setup_channel(const uint64_t channel_payload_buf_size)
{
    // Check if channel_payload_buf_size is valid
    channel_local_dequeue_idx_h     = 0;
    channel_payload_buf_size_h      = 1;
    channel_payload_buf_size_log2_h = 0;
    while (channel_payload_buf_size_h < channel_payload_buf_size)
    {
        channel_payload_buf_size_h *= 2;
        channel_payload_buf_size_log2_h++;
    }
    if (channel_payload_buf_size_h != channel_payload_buf_size)
    {
        fprintf(stderr, "channel_payload_buf_size must be a power of 2!\n");
        return;
    }

    // Initialize CUDA device and context
    CUDA_CHECK( cuInit(0) );
    CUDA_CHECK( cuDeviceGet(&dev0, 0) );
    CUDA_CHECK( cuCtxCreate(&ctx0, dev0, 0) );

    // Allocate page-locked host memory for channel messages
    size_t channel_payload_buf_bytes = sizeof(msg_payload_s) * channel_payload_buf_size_h;
    CUDA_CHECK( cuMemAllocHost((void **) &channel_payload_buf_h, channel_payload_buf_bytes) );
    CUDA_CHECK( cuMemAllocHost((void **) &channel_enqueue_idx_h, sizeof(uint64_t)) );
    CUDA_CHECK( cuMemAllocHost((void **) &channel_dequeue_idx_h, sizeof(uint64_t)) );
    *channel_enqueue_idx_h = 0;
    *channel_dequeue_idx_h = 0;
    // Initially correct payload flag is 1, fill it with 0 to mark all payload units are not ready
    memset(channel_payload_buf_h, 0, channel_payload_buf_bytes);

    // Map page-locked channel_buf_h and channel_idx_h pointer to GPU pointer
    uint64_t *tmp_payload_buf_dptr;
    uint64_t *tmp_enqueue_idx_dptr;
    uint64_t *tmp_dequeue_idx_dptr;
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &tmp_payload_buf_dptr, channel_payload_buf_h, 0) );
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &tmp_enqueue_idx_dptr, channel_enqueue_idx_h, 0) );
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &tmp_dequeue_idx_dptr, channel_dequeue_idx_h, 0) );

    // Copy GPU pointer to GPU memory
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_payload_buf_d, &tmp_payload_buf_dptr, 
        sizeof(uint64_t *), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_enqueue_idx_d, &tmp_enqueue_idx_dptr, 
        sizeof(uint64_t *), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_dequeue_idx_d, &tmp_dequeue_idx_dptr, 
        sizeof(uint64_t *), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_payload_buf_size_d, (const void*) &channel_payload_buf_size_h, 
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_payload_buf_size_log2_d, (const void*) &channel_payload_buf_size_log2_h, 
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        channel_local_dequeue_idx_d, (const void*) &channel_local_dequeue_idx_h, 
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice
    ));
}

void *host_daemon(void *param)
{
    uint64_t *param_  = (uint64_t *) param;
    uint64_t n_thread = param_[0];
    uint64_t channel_local_enqueue_idx_h;

    double result = 0.0;

    while (channel_local_dequeue_idx_h < n_thread)
    {
        channel_local_enqueue_idx_h = *((volatile uint64_t *) channel_enqueue_idx_h);
        //printf("[DEBUG] %llu --> %llu\n", channel_local_dequeue_idx_h, channel_local_enqueue_idx_h);
        for (; channel_local_dequeue_idx_h < channel_local_enqueue_idx_h; channel_local_dequeue_idx_h++)
        {
            // Get the correct flag for current dequeue index
            uint64_t dequeue_flag  = get_cyclic_queue_flag(channel_payload_buf_size_log2_h, channel_local_dequeue_idx_h);
            uint8_t  dequeue_flag8 = (uint8_t) dequeue_flag;

            // Wait the correct flag in each payload unit
            uint64_t payload_idx = channel_local_dequeue_idx_h & (channel_payload_buf_size_h - 1);
            uint64_t *payload_base_ptr = channel_payload_buf_h + payload_idx * MSG_PAYLOAD_LEN;
            msg_payload_unit_0_p payload_unit_0_ptr = (msg_payload_unit_0_p) payload_base_ptr;
            while (*((volatile uint8_t *) &payload_unit_0_ptr->flag) != dequeue_flag8);

            payload_base_ptr++;
            msg_payload_unit_1_p payload_unit_1_ptr = (msg_payload_unit_1_p) payload_base_ptr;
            while (*((volatile uint8_t *) &payload_unit_1_ptr->flag) != dequeue_flag8);

            // Extract payload data and assemble the original data
            uint64_t addend_buf = payload_unit_1_ptr->value_high;
            addend_buf = (addend_buf << 32) | payload_unit_0_ptr->value_low;

            double addend = *((double *) &addend_buf);
            result += addend;

            // Update the dequeue_idx
            *((volatile uint64_t *) channel_dequeue_idx_h) = channel_local_dequeue_idx_h;
        }
    }

    param_[1] = *((uint64_t *) &result);
    return NULL;
}

int main(int argc, char **argv)
{
    int grid_dim_x, block_dim_x, ret;
    printf("grid_dim_x, block_dim_x = ");
    ret = scanf("%d%d", &grid_dim_x, &block_dim_x);

    int n_thread = grid_dim_x * block_dim_x;

    const uint64_t channel_payload_buf_size = 128 * 1024;
    setup_channel(channel_payload_buf_size);

    uint64_t host_daemon_param[2] = {(uint64_t) n_thread, 0};
    pthread_t daemon_thread;
    ret = pthread_create(&daemon_thread, NULL, host_daemon, (void *) &host_daemon_param[0]);

    int64_t seed = (int64_t) (rand() + 1);
    test_kernel<<<grid_dim_x, block_dim_x>>>(1);
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
    printf("test_kernel completed\n");

    pthread_join(daemon_thread, NULL);
    double ref_result    = (double) (n_thread - 1) * (double) (n_thread) * 0.5;
    double daemon_result = *((double *) &host_daemon_param[1]);
    double result_relerr = fabs((ref_result - daemon_result) / ref_result);

    if (result_relerr < 1e-14) printf("host daemon result is correct\n");
    else printf("ref_result (%.15e) != host daemon result (%.15e)\n", ref_result, daemon_result);

    return 0;
}

