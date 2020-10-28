#include <pthread.h>

#include "msg_channel_host.hpp"
#include "msg_channel_sycl.hpp"

volatile int stop_daemon;
pthread_t daemon_thread;
size_t stop_idx;

void *msg_channel_host_daemon(void *param)
{
    uint64_t *msgchl_workbuf = (uint64_t *) param;
    uint64_t msgchl_buffer_size      = msgchl_workbuf[MSGCHL_BUFFER_SIZE_IDX];
    uint64_t msgchl_buffer_size_log2 = msgchl_workbuf[MSGCHL_BUFFER_SIZE_LOG2_IDX];
    uint64_t *msgchl_issue_idx_p     = msgchl_workbuf + MSGCHL_ISSUE_IDX_IDX;
    uint64_t *msgchl_process_idx_p   = msgchl_workbuf + MSGCHL_PROCESS_IDX_IDX;
    uint64_t *msgchl_buffer          = msgchl_workbuf + MSGCHL_BUFFER_START_IDX;

    uint64_t msgchl_process_idx_local = 0;

    int msgchl_sum = 0;

    while (!stop_daemon)
    {
        uint64_t msgchl_issue_idx_local = *((volatile uint64_t *) msgchl_issue_idx_p);

        for (; msgchl_process_idx_local < msgchl_issue_idx_local; msgchl_process_idx_local++)
        {
            uint64_t buffer_idx    = MSGCHL_WRAPPED_INDEX(msgchl_buffer_size, msgchl_process_idx_local);
            uint64_t process_flag0 = MSGCHL_CYCLIC_FLAG(msgchl_buffer_size_log2, msgchl_process_idx_local);
            uint8_t  process_flag  = (uint8_t) process_flag0;
            uint64_t *buffer_ptr   = msgchl_buffer + buffer_idx * MSG_PAYLOAD_LEN;

            msg_payload_unit_0_p unit0 = (msg_payload_unit_0_p) buffer_ptr;
            msg_payload_unit_1_p unit1 = (msg_payload_unit_1_p) (buffer_ptr + 1);
            while (*(volatile uint8_t *) &unit0->flag != process_flag);
            while (*(volatile uint8_t *) &unit1->flag != process_flag);
            msgchl_sum += unit0->value0 + unit1->value1;

            // Update the index of message processed on host
            *((volatile uint64_t *) msgchl_process_idx_p) = msgchl_process_idx_local;
        }

        if (msgchl_process_idx_local == stop_idx) stop_daemon = 1;
    }

    printf("msgchl  sum = %d\n", msgchl_sum);
    return NULL;
}

void msg_channel_host_daemon_start(uint64_t *msgchl_workbuf, const size_t n_workitem)
{
    stop_daemon = 0;
    stop_idx = n_workitem;
    int ret = pthread_create(&daemon_thread, NULL, msg_channel_host_daemon, (void *) msgchl_workbuf);
}

void msg_channel_host_daemon_wait_end()
{
    pthread_join(daemon_thread, NULL);
}