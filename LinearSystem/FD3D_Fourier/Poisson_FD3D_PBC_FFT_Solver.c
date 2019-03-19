#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

// See https://software.intel.com/en-us/mkl-developer-reference-c-fft-functions
#include "mkl_dfti.h"

#include "Poisson_FD3D_PBC_FFT_Solver.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

// Solve a Poisson equation -\nabla^2 u = f with period boundary condition
void Poisson_FD3D_PBC_FFT_Solver(
    const int nx, const int ny, const int nz, const int radius, 
    const double *w2_x, const double *w2_y, const double *w2_z, 
    double *f_rhs, double *u_sol
)
{
    double st, et;
    
    const int nd = nx * ny * nz;
    double *d_hat  = (double*) malloc(sizeof(double) * nd);
    double *cos_ix = (double*) malloc(sizeof(double) * nx * radius);
    double *cos_iy = (double*) malloc(sizeof(double) * ny * radius);
    double *cos_iz = (double*) malloc(sizeof(double) * nz * radius);
    double _Complex *f_hat = (double _Complex*) malloc(sizeof(double _Complex) * nd);
    assert(cos_ix != NULL && cos_iy != NULL && cos_iz != NULL && d_hat != NULL);
    assert(f_hat != NULL);
    
    // 1. Compute d_hat
    st = omp_get_wtime();
    // (1) Precompute all cosine values
    const int ix_s = (int) floor(-nx / 2.0) + 2;
    const int ix_e = (int) floor( nx / 2.0) + 1;
    const int iy_s = (int) floor(-ny / 2.0) + 2;
    const int iy_e = (int) floor( ny / 2.0) + 1;
    const int iz_s = (int) floor(-nz / 2.0) + 2;
    const int iz_e = (int) floor( nz / 2.0) + 1;
    for (int r = 0; r < radius; r++)
    {
        double tmp_x = 2.0 * M_PI * (double) (r+1) / (double) nx;
        double tmp_y = 2.0 * M_PI * (double) (r+1) / (double) ny;
        double tmp_z = 2.0 * M_PI * (double) (r+1) / (double) nz;
        for (int ix = ix_s; ix <= ix_e; ix++)
        {
            int ix1 = (ix - ix_s) * radius;
            cos_ix[ix1 + r] = cos((ix-1) * tmp_x) * w2_x[r+1];
        }
        for (int iy = iy_s; iy <= iy_e; iy++)
        {
            int iy1 = (iy - ix_s) * radius;
            cos_iy[iy1 + r] = cos((iy-1) * tmp_y) * w2_y[r+1];
        }
        for (int iz = iz_s; iz <= iz_e; iz++)
        {
            int iz1 = (iz - iz_s) * radius;
            cos_iz[iz1 + r] = cos((iz-1) * tmp_z) * w2_z[r+1];
        }
    }
    // (2) Compute d_hat using precomputed cosine values
    memset(d_hat, 0, sizeof(double) * nd);
    const double w2_diag = w2_x[0] + w2_y[0] + w2_z[0];
    // for iz = [1:iz_e, iz_s:0]
    for (int iz0 = 1; iz0 <= nz; iz0++)
    {
        int iz  = (iz0 > iz_e) ? iz0 - nz : iz0;
        int iz1 = (iz  - iz_s) * radius;
        // for iy = [1:iy_e, iy_s:0]
        for (int iy0 = 1; iy0 <= ny; iy0++)
        {
            int iy  = (iy0 > iy_e) ? iy0 - ny : iy0;
            int iy1 = (iy  - iy_s) * radius;
            // for ix = [1:ix_e, ix_s:0]
            for (int ix0 = 1; ix0 <= nx; ix0++)
            {
                int ix  = (ix0 > ix_e) ? ix0 - nx : ix0;
                int ix1 = (ix  - ix_s) * radius;
                
                double res = 0.0;
                for (int r = 0; r < radius; r++)
                    res += cos_ix[ix1 + r] + cos_iy[iy1 + r] + cos_iz[iz1 + r];
                res = -2.0 * res - w2_diag;
                int idx = (iz0-1) * nx * ny + (iy0-1) * nx + (ix0-1);
                d_hat[idx] = res;
            }
        }
    }
    et = omp_get_wtime();
    printf("Compute d_hat : %lf (s)\n", et - st);
    
    // 2. Compute f_hat
    // (1) Set up MKL FFT descriptor and perform FFT calculation
    st = omp_get_wtime();
    MKL_LONG status, dims[3] = {nz, ny, nx};
    DFTI_DESCRIPTOR_HANDLE mkl_fft_fwd_handle, mkl_fft_bwd_handle;
    status = DftiCreateDescriptor(&mkl_fft_fwd_handle, DFTI_DOUBLE, DFTI_REAL, 3, dims);
    status = DftiSetValue(mkl_fft_fwd_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(mkl_fft_fwd_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptor(mkl_fft_fwd_handle);
    status = DftiComputeForward(mkl_fft_fwd_handle, f_rhs, f_hat);
    status = DftiFreeDescriptor(&mkl_fft_fwd_handle);
    // (2) Unpack MKL FFT forward output
    unpackMKLDftiForward3DInplace(nz, ny, nx, f_hat);
    et = omp_get_wtime();
    printf("Compute f_hat : %lf (s)\n", et - st);
    
    // 3. Solve for u_hat, u_hat stored in f_hat
    st = omp_get_wtime();
    d_hat[0] = 1.0;
    for (int i = 0; i < nd; i++) f_hat[i] /= d_hat[i];
    f_hat[0] = 0.0;
    et = omp_get_wtime();
    printf("Compute u_hat : %lf (s)\n", et - st);
    
    // 4. Inverse FFT to get the result
    st = omp_get_wtime();
    double inv_nd = 1.0 / (double) nd;
    status = DftiCreateDescriptor(&mkl_fft_bwd_handle, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims);
    status = DftiSetValue(mkl_fft_bwd_handle, DFTI_BACKWARD_SCALE, inv_nd);
    status = DftiCommitDescriptor(mkl_fft_bwd_handle);
    status = DftiComputeBackward(mkl_fft_bwd_handle, f_hat);
    status = DftiFreeDescriptor(&mkl_fft_bwd_handle);    
    double sum_u = 0.0;
    for (int i = 0; i < nd; i++)
    {
        u_sol[i] = creal(f_hat[i]);
        sum_u += u_sol[i];
    }
    // ???
    sum_u = sum_u * inv_nd;
    for (int i = 0; i < nd; i++) u_sol[i] -= sum_u;
    et = omp_get_wtime();
    printf("Compute u_sol : %lf (s)\n", et - st);

    free(f_hat);
    free(d_hat);  
    free(cos_ix);
    free(cos_iy);
    free(cos_iz);
}