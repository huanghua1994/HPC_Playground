#ifndef __DRIVER_H__
#define __DRIVER_H__

#include <stdint.h>  // for uint32_t

typedef struct cuda_dev_state
{
    int      n_dev;
    int      dev_id;        // This is driver (not runtime) device id
    int      cu_device;     // CUdevice is defined as int
    void     *cu_context_p; // Pointer to a CUcontext type

    uint32_t host_hash;
    int      pcie_dev_id;
    int      pcie_bus_id;
    int      pcie_domain_id;
} cuda_dev_state_t, *cuda_dev_state_p;


#ifdef __cplusplus
extern "C" {
#endif

void init_cuda_dev_state(cuda_dev_state_p *state_);

void set_cuda_dev_id(cuda_dev_state_p state, const int dev_id);

void free_cuda_dev_state(cuda_dev_state_p *state_);

void check_cuda_dev_p2p(cuda_dev_state_p self, cuda_dev_state_p peer, int *can_p2p);

void get_cuda_ipc_mem_handle(void *dptr, int *handle_bytes, void **handle_);

void open_cuda_ipc_mem_handle(void **dptr, void *handle);

void close_cuda_ipc_mem_handle(void *dptr);

void get_cuda_device_cnt(int *cnt);

void set_cuda_device(const int dev_id);

void cuda_memcpy_h2d(void *hptr, void *dptr, const size_t bytes);

void cuda_memcpy_d2h(void *dptr, void *hptr, const size_t bytes);

void cuda_memcpy_auto(void *src, void *dst, const size_t bytes);

void alloc_cuda_mem(void **dptr_, const size_t bytes);

void free_cuda_mem(void *dptr);

void sync_cuda_device();

#ifdef __cplusplus
}
#endif

#endif 
