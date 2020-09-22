#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "CUDA_utils.h"
#include "msg_channel_cuda.cuh"

#define LO32B_MASK 0x00000000ffffffff
#define HI32B_MASK 0xffffffff00000000

// {issue, process}_idx are always increasing. Since it is 64 bit integer, 
// practically it will not overflow. 
__device__   uint64_t *ch_msg_buf_d;            // Array, size ch_msg_buf_size_d * MSG_PAYLOAD_LEN, 
                                                // message payload buffer, allocated on host, GPU writes, CPU reads
__device__   uint64_t *ch_issue_idx_d;          // Index of message issued on GPU, allocated on host, GPU writes, CPU reads
__device__   uint64_t *ch_process_idx_d;        // Index of message processed on host, allocated on host, CPU writes, GPU reads
__device__   uint64_t ch_process_idx_local_d;   // Only shared between GPU threads, to reduce access to *ch_msg_process_idx_d
__constant__ uint64_t ch_msg_buf_size_d;        // Size of message buffer
__constant__ uint64_t ch_msg_buf_size_log2_d;   // log2(size of message buffer)

CUdevice  dev0;
CUcontext ctx0;
uint64_t *ch_msg_buf_h;
uint64_t *ch_issue_idx_h;
uint64_t *ch_process_idx_h;
uint64_t ch_process_idx_local_h;
uint64_t ch_msg_buf_size_h;
uint64_t ch_msg_buf_size_log2_h;

__device__ inline void wait_until_added_ge_val(volatile uint64_t *addr, const uint64_t addend, const uint64_t val)
{
    uint64_t val_at_addr = *addr;
    while (val_at_addr + addend < val) val_at_addr = *addr;
}

__device__ inline void ch_msg_wait_buf_available(const uint64_t issue_idx)
{
    uint64_t process_idx = *((volatile uint64_t *) &ch_process_idx_local_d);
    if (process_idx + ch_msg_buf_size_d - 1 < issue_idx)
    {
        wait_until_added_ge_val(ch_process_idx_d, ch_msg_buf_size_d - 1, issue_idx);
        process_idx = *((volatile uint64_t *) ch_process_idx_d);
        atomicMax((unsigned long long int *) &ch_process_idx_local_d, process_idx);
        __threadfence_system();
    }
}

__device__ void ch_issue_double_msg(const double msg)
{
    // Get channel message buffer issue index and wait until it is safe to use
    __threadfence();
    uint64_t issue_idx = atomicAdd((unsigned long long int *) ch_issue_idx_d, 1);
    uint64_t write_idx = CHANNEL_WRAPPED_INDEX(ch_msg_buf_size_d, issue_idx);
    ch_msg_wait_buf_available(issue_idx);

    // Assemble message payload
    uint64_t msg_buf        = *((uint64_t *) &msg);
    uint64_t issue_flag     = CHANNEL_CYCLIC_FLAG(ch_msg_buf_size_log2_d, issue_idx);
    uint64_t payload_unit_0 = ((msg_buf << 32)        | issue_flag);
    uint64_t payload_unit_1 = ((msg_buf & HI32B_MASK) | issue_flag);

    // Write message payload to channel message buffer
    uint64_t *write_ptr = ch_msg_buf_d + write_idx * MSG_PAYLOAD_LEN;
    *((volatile uint64_t *) write_ptr) = payload_unit_0;
    write_ptr++;
    *((volatile uint64_t *) write_ptr) = payload_unit_1;
}

void msg_channel_cuda_setup(const uint64_t ch_msg_buf_size)
{
    // Check if ch_msg_buf_size is valid
    ch_process_idx_local_h = 0;
    ch_msg_buf_size_h      = 1;
    ch_msg_buf_size_log2_h = 0;
    while (ch_msg_buf_size_h < ch_msg_buf_size)
    {
        ch_msg_buf_size_h *= 2;
        ch_msg_buf_size_log2_h++;
    }
    if (ch_msg_buf_size_h != ch_msg_buf_size)
    {
        fprintf(stderr, "ch_msg_buf_size must be a power of 2!\n");
        return;
    }

    // Initialize CUDA device and context
    CUDA_CHECK( cuInit(0) );
    CUDA_CHECK( cuDeviceGet(&dev0, 0) );
    CUDA_CHECK( cuCtxCreate(&ctx0, 0, dev0) );

    // Allocate page-locked host memory for channel messages
    size_t ch_msg_buf_bytes = sizeof(msg_payload_t) * ch_msg_buf_size_h;
    CUDA_CHECK( cuMemAllocHost((void **) &ch_msg_buf_h,     ch_msg_buf_bytes) );
    CUDA_CHECK( cuMemAllocHost((void **) &ch_issue_idx_h,   sizeof(uint64_t)) );
    CUDA_CHECK( cuMemAllocHost((void **) &ch_process_idx_h, sizeof(uint64_t)) );
    *ch_issue_idx_h   = 0;
    *ch_process_idx_h = 0;
    // Initially correct payload flag is 1, fill it with 0 to mark all payload units are not ready
    memset(ch_msg_buf_h, 0, ch_msg_buf_bytes);

    // Map page-locked channel_buf_h and channel_idx_h pointer to GPU pointer
    uint64_t *ch_msg_buf_dptr;
    uint64_t *ch_msg_issue_idx_dptr;
    uint64_t *ch_msg_process_idx_dptr;
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &ch_msg_buf_dptr, ch_msg_buf_h, 0) );
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &ch_msg_issue_idx_dptr, ch_issue_idx_h, 0) );
    CUDA_CHECK( cuMemHostGetDevicePointer((CUdeviceptr *) &ch_msg_process_idx_dptr, ch_process_idx_h, 0) );

    // Copy GPU pointer to GPU memory
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        ch_msg_buf_d, &ch_msg_buf_dptr, 
        sizeof(uint64_t *), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        ch_issue_idx_d, &ch_msg_issue_idx_dptr, 
        sizeof(uint64_t *), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        ch_process_idx_d, &ch_msg_process_idx_dptr, 
        sizeof(uint64_t *), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        ch_msg_buf_size_d, (const void*) &ch_msg_buf_size_h, 
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        ch_msg_buf_size_log2_d, (const void*) &ch_msg_buf_size_log2_h, 
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice
    ));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(
        ch_process_idx_local_d, (const void*) &ch_process_idx_local_h, 
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice
    ));
}
