#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "CUDA_utils.h"
#include "driver.h"

// sdbm hash algorithm
uint32_t get_hostname_hash_sdbm()
{
    uint32_t result = 0;
    char hostname[1024];
    int ret = gethostname(hostname, 1024);
    for (int c = 0; hostname[c] != '\0'; c++)
        result = hostname[c] + (result << 6) + (result << 16) - result;
    return result;
}

void init_cuda_dev_state(cuda_dev_state_p *state_)
{
    cuda_dev_state_p state = (cuda_dev_state_p) malloc(sizeof(cuda_dev_state_t));

    CUDA_CHECK( cuInit(0) );
    CUDA_CHECK( cuDeviceGetCount(&state->n_dev) );
    state->dev_id         = -1;
    state->cu_device      = -1;
    state->cu_context_p   = malloc(sizeof(CUcontext));
    state->host_hash      = 0;
    state->pcie_dev_id    = -1;
    state->pcie_bus_id    = -1;
    state->pcie_domain_id = -1;

    *state_ = state;
}

void set_cuda_dev_id(cuda_dev_state_p state, const int dev_id)
{
    assert(state != NULL);
    if (dev_id >= state->n_dev || dev_id < 0)
    {
        fprintf(stderr, "Cannot set target device to %d: valid range [0, %d]\n", dev_id, state->n_dev);
        return;
    }

    state->dev_id = dev_id;
    CUDA_RUNTIME_CHECK( cudaSetDevice(dev_id) );
    CUDA_CHECK( cuCtxGetCurrent((CUcontext *) state->cu_context_p) );
    CUDA_CHECK( cuCtxGetDevice(&state->cu_device) );

    state->host_hash = get_hostname_hash_sdbm();
    CUDA_CHECK( cuDeviceGetAttribute(&state->pcie_dev_id,    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, state->cu_device) );
    CUDA_CHECK( cuDeviceGetAttribute(&state->pcie_bus_id,    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,    state->cu_device) );
    CUDA_CHECK( cuDeviceGetAttribute(&state->pcie_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, state->cu_device) );
}

void free_cuda_dev_state(cuda_dev_state_p *state_)
{
    cuda_dev_state_p state = *state_;
    if (state == NULL) return;
    free(state->cu_context_p);
    free(state);
    *state_ = NULL;
}

void check_cuda_dev_p2p(const int self_dev_id, const int peer_dev_id, int *can_p2p)
{
    CUDA_RUNTIME_CHECK( cudaDeviceCanAccessPeer(can_p2p, self_dev_id, peer_dev_id) );
    if (*can_p2p) CUDA_RUNTIME_CHECK( cudaDeviceEnablePeerAccess(peer_dev_id, 0) );
}

void get_cuda_ipc_mem_handle(void *dptr, int *handle_bytes, void **handle_)
{
    *handle_bytes = sizeof(cudaIpcMemHandle_t);
    cudaIpcMemHandle_t *handle_p = (cudaIpcMemHandle_t *) malloc(sizeof(cudaIpcMemHandle_t));
    CUDA_RUNTIME_CHECK( cudaIpcGetMemHandle(handle_p, dptr) );
    *handle_ = handle_p;
}

void open_cuda_ipc_mem_handle(void **dptr, void *handle)
{
    cudaIpcMemHandle_t handle1;
    memcpy(&handle1, handle, sizeof(cudaIpcMemHandle_t));
    CUDA_RUNTIME_CHECK( cudaIpcOpenMemHandle(dptr, handle1, cudaIpcMemLazyEnablePeerAccess) );
}

void close_cuda_ipc_mem_handle(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaIpcCloseMemHandle(dptr) );
}

void get_cuda_device_cnt(int *cnt)
{
    CUDA_RUNTIME_CHECK( cudaGetDeviceCount(cnt) );
}

void set_cuda_device(const int dev_id)
{
    CUDA_RUNTIME_CHECK( cudaSetDevice(dev_id) );
}

void cuda_memcpy_h2d(void *hptr, void *dptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr, hptr, bytes, cudaMemcpyHostToDevice) );
}

void cuda_memcpy_d2h(void *dptr, void *hptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(hptr, dptr, bytes, cudaMemcpyDeviceToHost) );
}

void cuda_memcpy_auto(void *src, void *dst, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) );
}

void alloc_cuda_mem(void **dptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMalloc(dptr_, bytes) );
}

void free_cuda_mem(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaFree(dptr) );
}

void sync_cuda_device()
{
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
}
