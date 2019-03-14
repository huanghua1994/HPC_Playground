#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#include "mkl_dfti.h"
#include "unpackMKLDftiForward.h"

/*
MATLAB code:
x = reshape(mod([1 : nx * ny * nz], 7), [nx, ny, nz]);
x_hat = fftn(x);
Then, x_hat(i, j, k) should equals to y[(k-1)*nx*ny + (j-1)*nx + (i-1)] in this code.
*/

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