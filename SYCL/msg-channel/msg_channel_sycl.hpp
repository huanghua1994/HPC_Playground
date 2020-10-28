#ifndef __MSG_CHANNEL_SYCL_HPP__
#define __MSG_CHANNEL_SYCL_HPP__

#include "../common/sycl_utils.hpp"

typedef enum 
{
    MSGCHL_BUFFER_SIZE_IDX = 0,
    MSGCHL_BUFFER_SIZE_LOG2_IDX,
    MSGCHL_ISSUE_IDX_IDX,
    MSGCHL_PROCESS_IDX_IDX,
    MSGCHL_BUFFER_START_IDX
} msgchl_workbuf_idx_t;

#define MSG_PAYLOAD_LEN 2

#define MSGCHL_CYCLIC_FLAG(log2_size, index)   ((uint8_t) (!((index >> log2_size) & 1)))
#define MSGCHL_WRAPPED_COUNT(log2_size, index) ((index) >> (log2_size))
#define MSGCHL_WRAPPED_INDEX(size, index)      ((index) & ((size) - 1) )  // size must be a power of 2!

typedef struct msg_payload
{
    uint64_t data[MSG_PAYLOAD_LEN];
} msg_payload_t, *msg_payload_p;

// high ---------------> low
//      32 |       24 |    8
//  value0 | reserved | flag 
typedef struct msg_payload_unit_0
{
    volatile uint8_t flag;
    uint8_t  resv0;
    uint16_t resv1;
    uint32_t value0;
} msg_payload_unit_0_t, *msg_payload_unit_0_p;

// high --------------> low
//    32  |       24 |    8
// value1 | reserved | flag 
typedef struct msg_payload_unit_1
{
    volatile uint8_t flag;
    uint8_t  resv0;
    uint16_t resv1;
    uint32_t value1;
} msg_payload_unit_1_t, *msg_payload_unit_1_p;

void msgchl_setup(sycl::queue &q, uint64_t msgchl_buffer_size, uint64_t **msgchl_workbuf_);

void test_msgchl_kernel(sycl::queue &q, uint64_t *msgchl_workbuf, const size_t n_workitem);

#endif
