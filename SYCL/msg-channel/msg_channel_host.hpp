#ifndef __MSG_CHANNEL_HOST_HPP__
#define __MSG_CHANNEL_HOST_HPP__

#include <stdint.h>

void msg_channel_host_daemon_start(uint64_t *msgchl_workbuf, const size_t n_workitem);

void msg_channel_host_daemon_wait_end();

#endif
