#ifndef __CUDA_PROXY_H__
#define __CUDA_PROXY_H__

// Wrap up some CUDA operations for non-CUDA modules

typedef struct cuda_dev_state
{
    int   n_dev;        // Number of CUDA devices
    int   dev_id;       // Driver (!= runtime ?) device id
    int   cu_dev;       // CUdevice is defined as int
    int   host_hash;    // Host name hash
    void  *cu_ctx_p;    // Pointer to a CUcontext type
} cuda_dev_state_t, *cuda_dev_state_p;


#ifdef __cplusplus
extern "C" {
#endif

// Get current process MPI intra-node / global rank, number of intra-node / global 
// processes from environment variables (before initializing MPI)
int  get_mpi_local_rank_env();
int  get_mpi_local_size_env();
int  get_mpi_global_rank_env();
int  get_mpi_global_size_env();

void cuda_init_dev_state(cuda_dev_state_p *state_);

void cuda_set_dev_id(cuda_dev_state_p state, const int dev_id);

void cuda_set_rt_dev_id(const int dev_id);

void cuda_free_dev_state(cuda_dev_state_p *state_);

int  cuda_check_dev_p2p(
    const int self_hash, const int self_dev_id,
    const int peer_hash, const int peer_dev_id
);

void cuda_get_ipc_mem_handle(void *dptr, int *handle_bytes, void **handle_);

void cuda_open_ipc_mem_handle(void **dptr, void *handle);

void cuda_close_ipc_mem_handle(void *dptr);

void cuda_memcpy_h2d(void *hptr, void *dptr, const size_t bytes);

void cuda_memcpy_d2h(void *dptr, void *hptr, const size_t bytes);

void cuda_memcpy_d2d(void *dptr_src, void *dptr_dst, const size_t bytes);

void cuda_memcpy_auto(void *src, void *dst, const size_t bytes);

void cuda_malloc_dev(void **dptr_, const size_t bytes);

void cuda_malloc_host(void **hptr_, const size_t bytes);

void cuda_memset_dev(void *dptr, const int value, const size_t bytes);

void cuda_memset_host(void *hptr, const int value, const size_t bytes);

void cuda_free_dev(void *dptr);

void cuda_free_host(void *hptr);

void cuda_device_sync();

void cuda_stream_sync(void *stream_p);

void cuda_print_last_error(const char *file, const int line);

#ifdef __cplusplus
}
#endif

#endif 
