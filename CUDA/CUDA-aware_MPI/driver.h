#ifndef __DRIVER_H__
#define __DRIVER_H__

#ifdef __cplusplus
extern "C" {
#endif

int get_gpu_device_cnt();

void set_gpu_device(const int dev_id);

void memcpy_h2d(void *hptr, void *dptr, const size_t bytes);

void memcpy_d2h(void *dptr, void *hptr, const size_t bytes);

void alloc_gpu_mem(void **dptr_, const size_t bytes);

void free_gpu_mem(void *dptr);

void sync_gpu_device();

#ifdef __cplusplus
}
#endif

#endif 
