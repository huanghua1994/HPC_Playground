#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>
#include "CUDA_Utils.h"

extern "C" {
#include "stencil_3axis_cuda_kernel.cuh"
}

extern "C"
void stencil_3axis_cuda(
    const double *x0, const int radius, const double *stencil_coefs, 
    const int nx, const int ny, const int nz, double *x1, const int ntest
)
{
    if (radius != RADIUS)
    {
        printf("FATAL: CUDA stencil kernel uses radius = %d, input radius = %d\n", RADIUS, radius);
        assert(RADIUS == radius);
    }
    
    const int nxyz  = nx * ny * nz;
    const int nx_ex = nx + 2 * radius;
    const int ny_ex = ny + 2 * radius;
    const int nz_ex = nz + 2 * radius;
    const int nxyz_ex = nx_ex * ny_ex * nz_ex;
    const int radius1 = radius + 1;
    
    // Allocate memory on device
    double *cu_x0, *cu_x1;
    cudaCheck( cudaMalloc(&cu_x0,   sizeof(double) * nxyz_ex) );
    cudaCheck( cudaMalloc(&cu_x1,   sizeof(double) * nxyz)    );
    
    // Copy input domain and coefficients to device
    cudaCheck( cudaMemcpy(cu_x0, x0, sizeof(double) * nxyz_ex, cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpyToSymbol(cu_coef, stencil_coefs, sizeof(double) * radius1)    );
    
    // Setup kernel parameters
    dim3 dim_block, dim_grid;
    dim_block.x = X_BLK_SIZE;
    dim_block.y = Y_BLK_SIZE;
    dim_grid.x  = (unsigned int) ceil((double) nx / (double) X_BLK_SIZE);
    dim_grid.y  = (unsigned int) ceil((double) ny / (double) Y_BLK_SIZE);
    printf("CUDA kernel: block size = %d * %d, grid size = %d * %d\n", 
            dim_block.x, dim_block.y, dim_grid.x, dim_grid.y);
    
    // Test the kernel
    cudaEvent_t st, et;
    double kernel_s;
    double GPoints = (double) nxyz / 1000000000.0;
    double GFlops  = GPoints * (7.0 * radius + 1);
    float kernel_ms;
    cudaCheck( cudaEventCreate(&st) );
    cudaCheck( cudaEventCreate(&et) );
    for (int i = 0; i < ntest; i++)
    {
        cudaCheck( cudaEventRecord(st, 0) );
        stencil_3axis_cuda_kernel<<<dim_grid, dim_block>>>(nx, ny, nz, cu_x0, cu_x1);
        cudaCheck( cudaEventRecord(et, 0) );
        cudaCheck( cudaEventSynchronize(et) );
        cudaEventElapsedTime(&kernel_ms, st, et);
        kernel_s = (double) kernel_ms * 1.0e-3;
        printf("CUDA kernel finished, used time = %lf (ms), %lf GPoint/s, %lf GFlops\n", 
               kernel_s * 1000.0, GPoints / kernel_s, GFlops / kernel_s);
    }
    printf("\n");
    
    // Copy results to host
    cudaCheck( cudaMemcpy(x1, cu_x1, sizeof(double) * nxyz, cudaMemcpyDeviceToHost) );
    
    // Free CUDA resources
    cudaCheck( cudaFree(cu_x0)   );
    cudaCheck( cudaFree(cu_x1)   );
}
