#ifndef __MSG_CHANNEL_CUDA_CUH__
#define __MSG_CHANNEL_CUDA_CUH__

#include <stdint.h>  // For uint*_t

#define MSG_PAYLOAD_LEN 2

typedef struct msg_payload
{
    uint64_t data[MSG_PAYLOAD_LEN];
} msg_payload_t, *msg_payload_p;

// high ------------------> low
//         32 |       24 |    8
//  value_low | reserved | flag 
typedef struct msg_payload_unit_0
{
    volatile uint8_t flag;
    uint8_t  resv0;
    uint16_t resv1;
    uint32_t value_low;
} msg_payload_unit_0_t, *msg_payload_unit_0_p;

// high ------------------> low
//        32  |       24 |    8
// value_high | reserved | flag 
typedef struct msg_payload_unit_1
{
    volatile uint8_t flag;
    uint8_t  resv0;
    uint16_t resv1;
    uint32_t value_high;
} msg_payload_unit_1_t, *msg_payload_unit_1_p;

#define CHANNEL_CYCLIC_FLAG(log2_size, index) ((uint8_t) (!((index >> log2_size) & 1)))
#define CHANNEL_WRAPPED_INDEX(size, index)    ((index) & ((size) - 1) )  // size must be a power of 2!

extern uint64_t *ch_msg_buf_h;
extern uint64_t *ch_issue_idx_h;
extern uint64_t *ch_process_idx_h;
extern uint64_t ch_process_idx_local_h;
extern uint64_t ch_msg_buf_size_h;
extern uint64_t ch_msg_buf_size_log2_h;

__device__ void ch_issue_double_msg(const double msg);

#ifdef __cplusplus
extern "C" {
#endif

void msg_channel_cuda_setup(const uint64_t ch_msg_buf_size);

#ifdef __cplusplus
}
#endif

#endif
