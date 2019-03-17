#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>
#include "CUDA_Utils.h"

#include "FD3D_CG_cuda.h"

#include "FD3D_CG_cuda_kernels.cuh"

void FD3D_CG_cuda(
    const int nx, const int ny, const int nz, 
    const double res_tol, const int max_iter, const double *b,
    const double *stencil_coefs, double *x
)
{
    const int nxyz  = nx * ny * nz;
    const int nx_ex = nx + 2 * RADIUS;
    const int ny_ex = ny + 2 * RADIUS;
    const int nz_ex = nz + 2 * RADIUS;
    const int nxyz_ex = nx_ex * ny_ex * nz_ex;
    const int radius1 = RADIUS + 1;
    
    // Allocate memory on device
    double *cu_x_ex, *cu_x, *cu_b, *cu_buff;
    cudaCheck( cudaMalloc(&cu_x_ex, sizeof(double) * nxyz_ex)  );
    cudaCheck( cudaMalloc(&cu_x,    sizeof(double) * nxyz)     );
    cudaCheck( cudaMalloc(&cu_b,    sizeof(double) * nxyz)     );
    cudaCheck( cudaMalloc(&cu_buff, sizeof(double) * nxyz * 4) );
    
    // Copy input domain and coefficients to device
    cudaCheck( cudaMemcpy(cu_x, x, sizeof(double) * nxyz, cudaMemcpyHostToDevice)  );
    cudaCheck( cudaMemcpy(cu_b, b, sizeof(double) * nxyz, cudaMemcpyHostToDevice)  );
    cudaCheck( cudaMemcpyToSymbol(cu_coef, stencil_coefs, sizeof(double) * radius1) );
    
    // Test the CG kernel
    cudaEvent_t st, et;
    double kernel_s = 0.0;
    float kernel_ms;
    cudaCheck( cudaEventCreate(&st) );
    cudaCheck( cudaEventCreate(&et) );
    cudaCheck( cudaEventRecord(st, 0) );
    cu_CG<<<1, 1>>>(nx, ny, nz, res_tol, max_iter, cu_b, cu_x_ex, cu_x, cu_buff);
    cudaCheckAfterCall();
    cudaCheck( cudaEventRecord(et, 0) );
    cudaCheck( cudaEventSynchronize(et) );
    cudaEventElapsedTime(&kernel_ms, st, et);
    kernel_s = (double) kernel_ms * 1.0e-3;
    printf("used time = %lf (s)\n", kernel_s);
    
    // Copy results to host
    cudaCheck( cudaMemcpy(x, cu_x, sizeof(double) * nxyz, cudaMemcpyDeviceToHost) );
    
    // Free CUDA resources
    cudaCheck( cudaFree(cu_x_ex) );
    cudaCheck( cudaFree(cu_x)    );
    cudaCheck( cudaFree(cu_b)    );
    cudaCheck( cudaFree(cu_buff) );
}
