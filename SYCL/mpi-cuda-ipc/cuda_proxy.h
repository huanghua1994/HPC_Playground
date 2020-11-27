#ifndef __CUDA_PROXY_H__
#define __CUDA_PROXY_H__

#ifdef __cplusplus
extern "C" {
#endif

int  cuda_get_device();

int  cuda_check_dev_p2p(const int self_dev_id, const int peer_dev_id);

void cuda_get_ipc_mem_handle(void *dptr, int *handle_bytes, void **handle_);

void cuda_open_ipc_mem_handle(void **dptr, void *handle);

void cuda_close_ipc_mem_handle(void *dptr);

#ifdef __cplusplus
}
#endif

#endif
