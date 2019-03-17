#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#include "cufftw.h"
#include "cuComplex.h"

#include "CUDA_Utils.h"

/*
MATLAB code:
x = reshape(mod([1 : nx * ny * nz], 7), [nx, ny, nz]);
x_hat = fftn(x);
Then, x_hat(i, j, k) should equals to y[(k-1)*nx*ny + (j-1)*nx + (i-1)] in this code.
*/

__global__ void cufftUnpackForward3D(const int nx, const int ny, const int nz, const int semi_nx, const int nxny, cufftDoubleComplex *y)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x + semi_nx;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int dst_z = blockIdx.z;
    const int src_x = nx - dst_x;
    const int src_y = (dst_y == 0) ? 0 : (ny - dst_y);
    const int src_z = (dst_z == 0) ? 0 : (nz - dst_z);
    if (dst_x < nx && dst_y < ny && dst_z < nz)
        y[dst_z * nxny + dst_y * nx + dst_x] = cuConj(y[src_z * nxny + src_y * nx + src_x]);
}

// Reference:
// 1. https://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout
// 2. https://docs.nvidia.com/cuda/cufft/index.html#function-cufftplanmany
void test_cuFFT_Forward(const int nx, const int ny, const int nz, double *x, double _Complex *y)
{
    cufftHandle        fwd_plan;
    cufftDoubleReal    *cu_x;
    cufftDoubleComplex *cu_y;
    int rank  = 3;
    int batch = 1;
    int nxyz  = nx * ny * nz;
    // Data layout:
    // data[b * nxyz + ((z * embed[1] + y) * embed[2] + x) * stride]
    // b is the index of input data in batch
    int dims[3]   = {nz, ny, nx};  // Dimensions, nz is the outmost loop, nx is the innermost loop
    int embeds[3] = { 1, ny, nx};  // Strides (lead dimensions) for x, z, y dimensions
    
    // Allocate input & output arrays on GPU
    cudaCheck( cudaMalloc(&cu_x, sizeof(cufftDoubleReal)    * nxyz) );
    cudaCheck( cudaMalloc(&cu_y, sizeof(cufftDoubleComplex) * nxyz) );
    cudaCheck( cudaMemcpy(cu_x, x, sizeof(double) * nxyz, cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemset(cu_y, 0, sizeof(cufftDoubleComplex) * nxyz) );
    
    // Create a FFT plan configuration and perform double --> double complex forward transform
    cuFFTCheck( cufftPlanMany(&fwd_plan, rank, dims, embeds, embeds[0], nxyz, embeds, embeds[0], nxyz, CUFFT_D2Z, batch) );
    cuFFTCheck( cufftExecD2Z(fwd_plan, cu_x, cu_y) );
    
    // Unpack the forward transform results
    dim3 dim_block, dim_grid;
    int semi_nx = nx / 2 + 1;
    int fill_nx = nx - semi_nx;
    dim_block.x = 16;
    dim_block.y = 4;
    dim_grid.x  = (fill_nx + dim_block.x - 1) / dim_block.x;
    dim_grid.y  = (ny      + dim_block.y - 1) / dim_block.y;
    dim_grid.z  = nz;
    cufftUnpackForward3D<<<dim_grid, dim_block>>>(nx, ny, nz, semi_nx, nx * ny, cu_y);
    cudaCheckAfterCall();
    
    // Copy output results back to host and free GPU arrays
    cudaCheck( cudaMemcpy(y, cu_y, sizeof(cufftDoubleComplex) * nxyz, cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(cu_x) );
    cudaCheck( cudaFree(cu_y) );
}

int main()
{
    while (1)
    {
        int nx, ny, nz, nd;
        int ret = scanf("%d%d%d", &nz, &ny, &nx);
        nd = nx * ny * nz;
        double *x = (double*) malloc(sizeof(double) * nd);
        double _Complex *y = (double _Complex*) malloc(sizeof(double _Complex) * nd);
        for (int i = 0; i < nd; i++) x[i] = (double) ((i + 1) % 7);
        
        test_cuFFT_Forward(nx, ny, nz, x, y);
        
        int cnt = 0;
        for (int iz = 0; iz < nz; iz++)
        {
            for (int iy = 0; iy < ny; iy++)
            {
                for (int ix = 0; ix < nx; ix++) 
                {
                    printf("%lf%+lfi ", creal(y[cnt]), cimag(y[cnt]));
                    cnt++;
                }
                printf("\n");
            }
            printf("--------------------\n");
        }
            
        free(x);
        free(y);
    }
}