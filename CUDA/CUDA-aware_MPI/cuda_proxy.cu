#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "CUDA_utils.h"
#include "cuda_proxy.h"

// sdbm hash algorithm
int get_hostname_hash_sdbm()
{
    int result = 0;
    char hostname[1024];
    int ret = gethostname(hostname, 1024);
    for (int c = 0; hostname[c] != '\0'; c++)
        result = hostname[c] + (result << 6) + (result << 16) - result;
    return result;
}

void cuda_init_dev_state(cuda_dev_state_p *state_)
{
    cuda_dev_state_p state = (cuda_dev_state_p) malloc(sizeof(cuda_dev_state_t));

    CUDA_CHECK( cuInit(0) );
    CUDA_CHECK( cuDeviceGetCount(&state->n_dev) );
    state->dev_id    = -1;
    state->cu_dev    = -1;
    state->cu_ctx_p  = malloc(sizeof(CUcontext));
    state->host_hash = get_hostname_hash_sdbm();

    *state_ = state;
}

void cuda_set_dev_id(cuda_dev_state_p state, const int dev_id)
{
    assert(state != NULL);
    if (dev_id >= state->n_dev || dev_id < 0)
    {
        fprintf(stderr, "Cannot set target device to %d: valid range [0, %d]\n", dev_id, state->n_dev);
        return;
    }

    state->dev_id = dev_id;
    CUDA_CHECK( cuInit(0) );
    CUDA_CHECK( cuDeviceGet(&state->cu_dev, dev_id) );
    CUDA_CHECK( cuCtxCreate((CUcontext *) state->cu_ctx_p, 0, state->cu_dev) );
}

void cuda_free_dev_state(cuda_dev_state_p *state_)
{
    cuda_dev_state_p state = *state_;
    if (state == NULL) return;
    free(state->cu_ctx_p);
    free(state);
    *state_ = NULL;
}

int  cuda_check_dev_p2p(
    const int self_hash, const int self_dev_id,
    const int peer_hash, const int peer_dev_id
)
{
    int can_p2p = 0;
    if (self_hash == peer_hash) CUDA_RUNTIME_CHECK( cudaDeviceCanAccessPeer(&can_p2p, self_dev_id, peer_dev_id) );
    return can_p2p;
}

void cuda_get_ipc_mem_handle(void *dptr, int *handle_bytes, void **handle_)
{
    *handle_bytes = sizeof(cudaIpcMemHandle_t);
    cudaIpcMemHandle_t *handle_p = (cudaIpcMemHandle_t *) malloc(sizeof(cudaIpcMemHandle_t));
    CUDA_RUNTIME_CHECK( cudaIpcGetMemHandle(handle_p, dptr) );
    *handle_ = handle_p;
}

void cuda_open_ipc_mem_handle(void **dptr, void *handle)
{
    CUDA_RUNTIME_CHECK( cudaIpcOpenMemHandle(dptr, *((cudaIpcMemHandle_t *) handle), cudaIpcMemLazyEnablePeerAccess) );
}

void cuda_close_ipc_mem_handle(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaIpcCloseMemHandle(dptr) );
}

void cuda_memcpy_h2d(void *hptr, void *dptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr, hptr, bytes, cudaMemcpyHostToDevice) );
}

void cuda_memcpy_d2h(void *dptr, void *hptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(hptr, dptr, bytes, cudaMemcpyDeviceToHost) );
}

void cuda_memcpy_d2d(void *dptr_src, void *dptr_dst, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr_dst, dptr_src, bytes, cudaMemcpyDeviceToDevice) );
}

void cuda_memcpy_auto(void *src, void *dst, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) );
}

void cuda_malloc_dev(void **dptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMalloc(dptr_, bytes) );
}

void cuda_malloc_host(void **hptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMallocHost(hptr_, bytes) );
}

void cuda_memset_dev(void *dptr, const int value, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemset(dptr, value, bytes) );    
}

void cuda_memset_host(void *hptr, const int value, const size_t bytes)
{
    memset(hptr, value, bytes); 
}

void cuda_free_dev(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaFree(dptr) );
}

void cuda_free_host(void *hptr)
{
    CUDA_RUNTIME_CHECK( cudaFree(hptr) );
}

void cuda_device_sync()
{
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
}

void cuda_stream_sync(void *stream_p)
{
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(*((cudaStream_t *) stream_p)) );
}

