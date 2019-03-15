#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#include "mkl_dfti.h"

/*
MATLAB code:
x = reshape(mod([1 : nx * ny * nz], 7), [nx, ny, nz]);
x_hat = fftn(x);
Then, x_hat(i, j, k) should equals to y[(k-1)*nx*ny + (j-1)*nx + (i-1)] in this code.
*/

// See: https://software.intel.com/sites/default/files/managed/42/5b/Unpack_result_of_Intel_Math_Kernel_Library_Fast_Fourier_Transform_routines_to_align_with_Matlab.pdf
// Unpack MKL 3D FFT CCE output format after 3D FFT forward transform
// Use for 2D or 1D output: set nz & ny = 1
// Input parameters:
//   n{z, y, x} : Data dimensions, z is the slowest running index and x is the fastest
//   data       : MKL FFT 3D forward transform output
// Output parameters:
//   data       : Unpacked FFT 3D forward transform result
void unpackMKLDftiForward3DInplace(const int nz, const int ny, const int nx, double _Complex *data)
{
    int semi_nx = nx / 2 + 1;
    int nxny = nx * ny;
    for (int dst_z = 0; dst_z < nz; dst_z++)
    {
        int src_z = (dst_z == 0) ? 0 : (nz - dst_z);
        for (int dst_y = 0; dst_y < ny; dst_y++)
        {
            int src_y = (dst_y == 0) ? 0 : (ny - dst_y);
            int src_offset = src_z * nxny + src_y * nx;
            int dst_offset = dst_z * nxny + dst_y * nx;
            for (int dst_x = semi_nx; dst_x < nx; dst_x++)
            {
                int src_x = nx - dst_x;
                data[dst_offset + dst_x] = conj(data[src_offset + src_x]);
            }
        }
    }
}

int main()
{
    while (1)
    {
        int nx, ny, nz, nd;
        scanf("%d%d%d", &nz, &ny, &nx);
        nd = nx * ny * nz;
        double *x = (double*) malloc(sizeof(double) * nd);
        double _Complex *y = (double _Complex*) malloc(sizeof(double _Complex) * nd);
        for (int i = 0; i < nd; i++) x[i] = (double) ((i + 1) % 7);
        
        MKL_LONG status, dims[3] = {nz, ny, nx};
        DFTI_DESCRIPTOR_HANDLE mkl_fft_fwd_handle;
        status = DftiCreateDescriptor(&mkl_fft_fwd_handle, DFTI_DOUBLE, DFTI_REAL, 3, dims);
        status = DftiSetValue(mkl_fft_fwd_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiSetValue(mkl_fft_fwd_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiCommitDescriptor(mkl_fft_fwd_handle);
        status = DftiComputeForward(mkl_fft_fwd_handle, x, y);
        status = DftiFreeDescriptor(&mkl_fft_fwd_handle);
        
        unpackMKLDftiForward3DInplace(nz, ny, nx, y);
        
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