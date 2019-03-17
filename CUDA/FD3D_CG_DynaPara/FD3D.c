#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "FD3D.h"

// Copy x to the extended domain x_ex
// Input parameters:
//    nx, ny, nz : Sizes of x, y, z direction
//    radius     : Radius of the stencil
//    x          : Values in the original domain
// Output parameters:
//    x_ex       : Values in the extended domain (boundary condition not handled)
static void FD3D_copy_inner(
    const int nx, const int ny, const int nz, const int radius,
    const double *x, double *x_ex
)
{
    const int nx_ex = nx + 2 * radius;
    const int ny_ex = ny + 2 * radius;
    const int nz_ex = nz + 2 * radius;
    size_t line_size = sizeof(double) * nx;
    #pragma omp parallel for collapse(2)
    for (int kp = radius; kp < nz + radius; kp++) 
    {
        for (int jp = radius; jp < ny + radius; jp++) 
        {
            int ex_offset =  kp           * nx_ex * ny_ex +  jp           * nx_ex + radius;
            int  x_offset = (kp - radius) * nx    * ny    + (jp - radius) * nx;
            double *x_ex_ptr = x_ex + ex_offset;
            const double *x_ptr = x +  x_offset;
            memcpy(x_ex_ptr, x_ptr, line_size);
        }
    }
}

// Kernel for calculating Laplacian * x
// For the input & output domain, z/x index is the slowest/fastest running index
// Input parameters:
//    x0               : Input domain with extended boundary 
//    radius           : Radius of the stencil (radius * 2 = stencil order)
//    stride_y         : Distance between x1(x, y, z) and x1(x, y+1, z)
//    stride_y_ex      : Distance between x0(x, y, z) and x0(x, y+1, z)
//    stride_z         : Distance between x1(x, y, z) and x1(x, y, z+1)
//    stride_z_ex      : Distance between x0(x, y, z) and x0(x, y, z+1)
//    [x_spos, x_epos) : X index range that will be computed in this kernel
//    [y_spos, y_epos) : Y index range that will be computed in this kernel
//    [z_spos, z_epos) : Z index range that will be computed in this kernel
//    coef             : Stencil coefficients, 3 * coef[0] is for the center element
// Output parameters:
//    x1               : Output domain with original boundary
static void stencil_3axis_thread(
    const double *x0,   const int radius, 
    const int stride_y, const int stride_y_ex, 
    const int stride_z, const int stride_z_ex,
    const int x_spos,   const int x_epos, 
    const int y_spos,   const int y_epos,
    const int z_spos,   const int z_epos,
    const double *coef, double *x1
)
{
    const double coef_0 = coef[0] * 3.0;
    for (int z = z_spos; z < z_epos; z++)
    {
        int iz = z + radius;
        for (int y = y_spos; y < y_epos; y++)
        {
            int iy = y + radius;
            int offset = z * stride_z + y * stride_y;
            int offset_ex = iz * stride_z_ex + iy * stride_y_ex + radius;
            
            #pragma omp simd
            for (int x = x_spos; x < x_epos; x++)
            {
                int idx = offset + x;
                int idx_ex = offset_ex + x;
                double res = coef_0 * x0[idx_ex];
                for (int r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    res += coef[r] * (x0[idx_ex + r]          + x0[idx_ex - r] +
                                      x0[idx_ex + stride_y_r] + x0[idx_ex - stride_y_r] +
                                      x0[idx_ex + stride_z_r] + x0[idx_ex - stride_z_r]);
                }
                x1[idx] = res;
            }
        }
    }
}

#define X_BLK_SIZE 16
#define Y_BLK_SIZE 8
#define Z_BLK_SIZE 16

int    nblocks;
int    *x_spos = NULL;
int    *y_spos = NULL;
int    *z_spos = NULL;
int    *x_epos = NULL;
int    *y_epos = NULL;
int    *z_epos = NULL;
double *x_ex   = NULL;
int nx, ny, nz, radius, nxyz;
int nx_ex, ny_ex, nz_ex, nxyz_ex;
int stride_y, stride_z, stride_y_ex, stride_z_ex;

// Set parameters for Laplacian operator
void FD3D_Laplacian_set_param(const int nx_, const int ny_, const int nz_, const int radius_)
{
    nx = nx_;
    ny = ny_;
    nz = nz_;
    radius = radius_;
    
    nxyz = nx * ny * nz;
    nx_ex = nx + 2 * radius;
    ny_ex = ny + 2 * radius;
    nz_ex = nz + 2 * radius;
    nxyz_ex = nx_ex * ny_ex * nz_ex;
    stride_y = nx;
    stride_z = nx * ny;
    stride_y_ex = nx_ex;
    stride_z_ex = nx_ex * ny_ex;
    
    int nblk_x   = (nx + X_BLK_SIZE - 1) / X_BLK_SIZE;
    int nblk_y   = (ny + Y_BLK_SIZE - 1) / Y_BLK_SIZE;
    int nblk_z   = (nz + Z_BLK_SIZE - 1) / Z_BLK_SIZE;
    int nblk_xy  = nblk_x * nblk_y;
    int nblk_xyz = nblk_xy * nblk_z;
    nblocks = nblk_xyz;

    size_t idxs_msize = sizeof(int) * nblk_xyz;
    free(x_spos);
    free(y_spos);
    free(z_spos);
    free(x_epos);
    free(y_epos);
    free(z_epos);
    x_spos = (int*) malloc(idxs_msize);
    y_spos = (int*) malloc(idxs_msize);
    z_spos = (int*) malloc(idxs_msize);
    x_epos = (int*) malloc(idxs_msize);
    y_epos = (int*) malloc(idxs_msize);
    z_epos = (int*) malloc(idxs_msize);
    for (int i = 0; i < nblk_xyz; i++)
    {
        int iblk_z = i / nblk_xy;
        int iblk_y = (i % nblk_xy) / nblk_x;
        int iblk_x = i % nblk_x;
        z_spos[i] = iblk_z * Z_BLK_SIZE;
        y_spos[i] = iblk_y * Y_BLK_SIZE;
        x_spos[i] = iblk_x * X_BLK_SIZE;
        z_epos[i] = (z_spos[i] + Z_BLK_SIZE > nz) ? nz : z_spos[i] + Z_BLK_SIZE;
        y_epos[i] = (y_spos[i] + Y_BLK_SIZE > ny) ? ny : y_spos[i] + Y_BLK_SIZE;
        x_epos[i] = (x_spos[i] + X_BLK_SIZE > nx) ? nx : x_spos[i] + X_BLK_SIZE;
    }
    
    free(x_ex);
    x_ex = (double*) malloc(sizeof(double) * nxyz_ex);
    // Use zero boundary condition
    #pragma omp parallel for
    for (int i = 0; i < nxyz_ex; i++) x_ex[i] = 0.0;
}

// Laplacian operator
void FD3D_Laplacian_MatVec(const double *x_in, double *x_out)
{
    FD3D_copy_inner(nx, ny, nz, radius, x_in, x_ex);
    
    #pragma omp parallel for schedule(static)
    for (int iblk = 0; iblk < nblocks; iblk++)
    {
        stencil_3axis_thread(
            x_ex, radius, stride_y, stride_y_ex, stride_z, stride_z_ex,
            x_spos[iblk], x_epos[iblk], 
            y_spos[iblk], y_epos[iblk], 
            z_spos[iblk], z_epos[iblk], 
            stencil_coefs, x_out
        );
    }
}