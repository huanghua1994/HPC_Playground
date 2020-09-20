#ifndef __TEST_KERNEL_H__
#define __TEST_KERNEL_H__

#ifdef __cplusplus
extern "C" {
#endif

void launch_test_kernel(const int grid_dim_x, const int block_dim_x);

#ifdef __cplusplus
}
#endif

#endif
