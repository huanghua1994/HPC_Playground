#ifndef __MSG_CHANNEL_HOST_H__
#define __MSG_CHANNEL_HOST_H__

#include <stdint.h>  // For uint*_t

#define HOST_DAEMON_PARAM_SIZE  4

extern uint64_t host_daemon_param[HOST_DAEMON_PARAM_SIZE];

#ifdef __cplusplus
extern "C" {
#endif

void msg_channel_setup(const uint64_t ch_msg_buf_size);

void msg_channel_stop();

#ifdef __cplusplus
}
#endif

#endif 

