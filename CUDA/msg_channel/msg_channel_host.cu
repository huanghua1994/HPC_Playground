#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

#include "CUDA_utils.h"
#include "msg_channel_cuda.cuh"
#include "msg_channel_host.h"

// In this test, host_daemon_param will be initialized by external functions
uint64_t  host_daemon_param[HOST_DAEMON_PARAM_SIZE];
pthread_t daemon_thread;

void *host_daemon(void *param)
{
    uint64_t *param_  = (uint64_t *) param;
    uint64_t n_thread = param_[0];
    uint64_t ch_issue_idx_local_h;

    double result = 0.0;

    while (ch_process_idx_local_h < n_thread)
    {
        ch_issue_idx_local_h = *((volatile uint64_t *) ch_issue_idx_h);
        //printf("[DEBUG] %llu --> %llu\n", ch_process_idx_local_h, ch_issue_idx_local_h);
        for (; ch_process_idx_local_h < ch_issue_idx_local_h; ch_process_idx_local_h++)
        {
            // Get the correct flag for current dequeue index
            uint64_t process_flag  = CHANNEL_CYCLIC_FLAG(ch_msg_buf_size_log2_h, ch_process_idx_local_h);
            uint8_t  process_flag8 = (uint8_t) process_flag;

            // Wait the correct flag in each payload unit
            uint64_t payload_idx = CHANNEL_WRAPPED_INDEX(ch_msg_buf_size_h, ch_process_idx_local_h);
            uint64_t *payload_base_ptr = ch_msg_buf_h + payload_idx * MSG_PAYLOAD_LEN;
            
            msg_payload_unit_0_p payload_unit_0_ptr = (msg_payload_unit_0_p) payload_base_ptr;
            while (*((volatile uint8_t *) &payload_unit_0_ptr->flag) != process_flag8);

            payload_base_ptr++;
            msg_payload_unit_1_p payload_unit_1_ptr = (msg_payload_unit_1_p) payload_base_ptr;
            while (*((volatile uint8_t *) &payload_unit_1_ptr->flag) != process_flag8);

            // Extract payload data and assemble the original data
            uint64_t addend_buf = payload_unit_1_ptr->value_high;
            addend_buf = (addend_buf << 32) | payload_unit_0_ptr->value_low;

            double addend = *((double *) &addend_buf);
            result += addend;

            // Update the dequeue_idx
            *((volatile uint64_t *) ch_process_idx_h) = ch_process_idx_local_h;
        }
    }

    param_[1] = *((uint64_t *) &result);
    return NULL;
}

void msg_channel_setup(const uint64_t ch_msg_buf_size)
{
    msg_channel_cuda_setup(ch_msg_buf_size);
    int ret = pthread_create(&daemon_thread, NULL, host_daemon, (void *) &host_daemon_param[0]);
}

void msg_channel_stop()
{
    // Signal the daemon thread using host_daemon_param
    // In this test, nothing need to be done here
    
    pthread_join(daemon_thread, NULL);
}
