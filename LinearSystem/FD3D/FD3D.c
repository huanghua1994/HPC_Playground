#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "FD3D.h"

// Copy a block from a 3D tensor to another 3D tensor, 
// z/x index is the slowest/fastest running index
// Input parameters:
//   src      : Pointer to the 1st element in the source 3D tensor
//   s_ldz    : Distance between src(x, y, z) and src(x, y, z+1)
//   s_ldy    : Distance between src(x, y, z) and src(x, y+1, z)
//   n{x,y,z} : Size of the block to be copied
//   d_ldz    : Distance between dst(x, y, z) and dst(x, y, z+1)
//   d_ldy    : Distance between dst(x, y, z) and dst(x, y+1, z)
// Output parameters:
//   dst : Pointer to the 1st element in the destination 3D tensor
static void copy_dbl_3D_block(
    const double *src, const int s_ldz, const int s_ldy, 
    const int nx,      const int ny,    const int nz,
    double *dst,       const int d_ldz, const int d_ldy
)
{
    size_t nx_msize = sizeof(double) * nx;
    #pragma omp parallel for collapse(2)
    for (int iz = 0; iz < nz; iz++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            const double *src_ptr = src + iz * s_ldz + iy * s_ldy;
            double *dst_ptr = dst + iz * d_ldz + iy * d_ldy;
            memcpy(dst_ptr, src_ptr, nx_msize);
        }
    }
}

// Kernel for calculating Laplacian * x
// For the input & output domain, z/x index is the slowest/fastest running index
// Input parameters:
//    x0               : Input domain with extended boundary 
//    FDn              : Radius of the stencil (FDn * 2 = stencil order)
//    stride_y         : Distance between x1(x, y, z) and x1(x, y+1, z)
//    stride_y_ex      : Distance between x0(x, y, z) and x0(x, y+1, z)
//    stride_z         : Distance between x1(x, y, z) and x1(x, y, z+1)
//    stride_z_ex      : Distance between x0(x, y, z) and x0(x, y, z+1)
//    [x_spos, x_epos) : X index range that will be computed in this kernel
//    [y_spos, y_epos) : Y index range that will be computed in this kernel
//    [z_spos, z_epos) : Z index range that will be computed in this kernel
//    {x, y, z}_coef   : Stencil coefficients
// Output parameters:
//    x1 : Output domain with original boundary
static void stencil_3axis_thread(
    const double *x0,   const int FDn, 
    const int stride_y, const int stride_y_ex, 
    const int stride_z, const int stride_z_ex,
    const int x_spos,   const int x_epos, 
    const int y_spos,   const int y_epos,
    const int z_spos,   const int z_epos,
    const double *x_coef, const double *y_coef, const double *z_coef,
    double *x1
)
{
    const double coef_0 = x_coef[0] + y_coef[0] + z_coef[0];
    for (int z = z_spos; z < z_epos; z++)
    {
        int iz = z + FDn;
        for (int y = y_spos; y < y_epos; y++)
        {
            int iy = y + FDn;
            int offset = z * stride_z + y * stride_y;
            int offset_ex = iz * stride_z_ex + iy * stride_y_ex + FDn;
            
            #pragma omp simd
            for (int x = x_spos; x < x_epos; x++)
            {
                int    idx    = offset + x;
                int    idx_ex = offset_ex + x;
                double res_x  = coef_0 * x0[idx_ex];
                double res_y  = 0.0;
                double res_z  = 0.0;
                for (int r = 1; r <= FDn; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    res_x += x_coef[r] * (x0[idx_ex + r]          + x0[idx_ex - r]);
                    res_y += y_coef[r] * (x0[idx_ex + stride_y_r] + x0[idx_ex - stride_y_r]);
                    res_z += z_coef[r] * (x0[idx_ex + stride_z_r] + x0[idx_ex - stride_z_r]);
                }
                x1[idx] = res_x + res_y + res_z;
            }
        }
    }
}

#define X_BLK_SIZE 16
#define Y_BLK_SIZE 8
#define Z_BLK_SIZE 16

static int    nblk;
static int    *x_spos = NULL;
static int    *y_spos = NULL;
static int    *z_spos = NULL;
static int    *x_epos = NULL;
static int    *y_epos = NULL;
static int    *z_epos = NULL;
static double *x_ex   = NULL;
static double *w2     = NULL;
static double *x_coef = NULL;
static double *y_coef = NULL;
static double *z_coef = NULL;
static int    FDn;
static int    nx,    ny,    nz,    nxyz;
static int    BCx,   BCy,   BCz;
static int    nx_ex, ny_ex, nz_ex, nxyz_ex;
static int    stride_y, stride_z, stride_y_ex, stride_z_ex;

static double factorial(const int n)
{
    double res = 1.0;
    for (int i = 2; i <= n; i++)
        res *= (double) i;
    return res;
}

// Set parameters for Laplacian operator
void FD3D_Laplacian_set_param(const double *cell_dims, const int *grid_sizes, const int *BCs, const int FDn_)
{
    FDn    = FDn_;
    nx     = grid_sizes[0];
    ny     = grid_sizes[1];
    nz     = grid_sizes[2];
    BCx    = BCs[0];
    BCy    = BCs[1];
    BCz    = BCs[2];
    
    nxyz        = nx * ny * nz;
    nx_ex       = nx + 2 * FDn;
    ny_ex       = ny + 2 * FDn;
    nz_ex       = nz + 2 * FDn;
    nxyz_ex     = nx_ex * ny_ex * nz_ex;
    stride_y    = nx;
    stride_z    = nx * ny;
    stride_y_ex = nx_ex;
    stride_z_ex = nx_ex * ny_ex;
    
    // 1. Calculate the finite difference weights
    w2 = (double *) malloc(sizeof(double) * (FDn + 1));
    assert(w2 != NULL);
    double c1 = factorial(FDn);  c1 *= c1;
    for (int k = 1; k <= FDn; k++)
    {
        double c2 = ((k + 1) % 2) ? -2.0 : 2.0;
        double c3 = 1.0 / (double) (k * k);
        double c4 = factorial(FDn - k);
        double c5 = factorial(FDn + k);
        w2[k]  = c1 * c2 * c3 / (c4 * c5);
        w2[0] -= 2.0 * c3;
    }
    
    // 2. Combine the FD weights with mesh space
    int nd = nx * ny * nz;
    double inv_dx2 = cell_dims[0] / (double) (nx - BCx);
    double inv_dy2 = cell_dims[1] / (double) (ny - BCy);
    double inv_dz2 = cell_dims[2] / (double) (nz - BCz);
    inv_dx2 = 1.0 / (inv_dx2 * inv_dx2);
    inv_dy2 = 1.0 / (inv_dy2 * inv_dy2);
    inv_dz2 = 1.0 / (inv_dz2 * inv_dz2);
    x_coef = (double*) malloc(sizeof(double) * (FDn + 1));
    y_coef = (double*) malloc(sizeof(double) * (FDn + 1));
    z_coef = (double*) malloc(sizeof(double) * (FDn + 1));
    assert(x_coef != NULL && y_coef != NULL && z_coef != NULL);
    for (int r = 0; r <= FDn; r++)
    {
        x_coef[r] = w2[r] * inv_dx2;
        y_coef[r] = w2[r] * inv_dy2;
        z_coef[r] = w2[r] * inv_dz2;
    }
    
    // 3. Partition the FD domain for OpenMP parallelization
    int nblk_x   = (nx + X_BLK_SIZE - 1) / X_BLK_SIZE;
    int nblk_y   = (ny + Y_BLK_SIZE - 1) / Y_BLK_SIZE;
    int nblk_z   = (nz + Z_BLK_SIZE - 1) / Z_BLK_SIZE;
    int nblk_xy  = nblk_x * nblk_y;
    int nblk_xyz = nblk_xy * nblk_z;
    nblk = nblk_xyz;

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
    
    // Use zero boundary condition by default, copy the periodic boundary later 
    #pragma omp parallel for
    for (int i = 0; i < nxyz_ex; i++) x_ex[i] = 0.0;
}

// Laplacian operator
void FD3D_Laplacian_MatVec(const double *x_in, double *x_out)
{
    double *dst;
    dst = x_ex + FDn * (stride_z_ex + stride_y_ex + 1);
    // x_in(0:nx-1, 0:ny-1, 0:nz-1) --> x_ex(FDn:nx+FDn-1, FDn:ny+FDn-1, FDn:nz+FDn-1)
    copy_dbl_3D_block(
        x_in, stride_z, stride_y, nx, ny, nz, 
        dst, stride_z_ex, stride_y_ex
    );
    
    if (BCx == 0)
    {
        // x_in(0:FDn-1, 0:ny-1, 0:nz-1) --> x_ex(FDn+nx:FDn+nx+FDn-1, FDn:ny+FDn-1, FDn:nz+FDn-1)
        const double *src0 = x_in;
        dst = x_ex + (FDn + nx) + FDn * stride_y_ex + FDn * stride_z_ex;
        copy_dbl_3D_block(
            src0, stride_z, stride_y, FDn, ny, nz, 
            dst, stride_z_ex, stride_y_ex
        );
        // x_in(nx-FDn:nx-1, 0:ny-1, 0:nz-1) --> x_ex(0:FDn-1, FDn:ny+FDn-1, FDn:nz+FDn-1)
        const double *src1 = x_in + (nx - FDn);
        dst = x_ex + FDn * stride_y_ex + FDn * stride_z_ex;
        copy_dbl_3D_block(
            src1, stride_z, stride_y, FDn, ny, nz, 
            dst, stride_z_ex, stride_y_ex
        );
    }
    
    if (BCy == 0)
    {
        // x_in(0:nx-1, 0:FDn-1, 0:nz-1) --> x_ex(FDn:nx+FDn-1, FDn+ny:FDn+ny+FDn-1, FDn:nz+FDn-1)
        const double *src0 = x_in;
        dst = x_ex + FDn + (FDn + ny) * stride_y_ex + FDn * stride_z_ex;
        copy_dbl_3D_block(
            src0, stride_z, stride_y, nx, FDn, nz, 
            dst, stride_z_ex, stride_y_ex
        );
        // x_in(0:nx-1, ny-FDn:ny-1, 0:nz-1) --> x_ex(FDn:nx+FDn-1, 0:FDn-1, FDn:nz+FDn-1)
        const double *src1 = x_in + (ny - FDn) * stride_y;
        dst = x_ex + FDn + FDn * stride_z_ex;
        copy_dbl_3D_block(
            src1, stride_z, stride_y, nx, FDn, nz, 
            dst, stride_z_ex, stride_y_ex
        );
    }
    
    if (BCz == 0)
    {
        // x_in(0:nx-1, 0:ny-1, 0:FDn-1) --> x_ex(FDn:nx+FDn-1, FDn:ny+FDn-1, FDn+nz:FDn+nz+FDn-1)
        const double *src0 = x_in;
        dst = x_ex + FDn + FDn * stride_y_ex + (FDn + nz) * stride_z_ex;
        copy_dbl_3D_block(
            src0, stride_z, stride_y, nx, ny, FDn, 
            dst, stride_z_ex, stride_y_ex
        );
        // x_in(0:nx-1, 0:ny-1, nz-FDn:nz-1) --> x_ex(FDn:nx+FDn-1, FDn:ny+FDn-1, 0:FDn-1)
        const double *src1 = x_in + (nz - FDn) * stride_z;
        dst = x_ex + FDn + FDn * stride_y_ex;
        copy_dbl_3D_block(
            src1, stride_z, stride_y, nx, ny, FDn, 
            dst, stride_z_ex, stride_y_ex
        );
    }
    
    #pragma omp parallel for schedule(static)
    for (int iblk = 0; iblk < nblk; iblk++)
    {
        stencil_3axis_thread(
            x_ex, FDn, stride_y, stride_y_ex, stride_z, stride_z_ex,
            x_spos[iblk], x_epos[iblk], 
            y_spos[iblk], y_epos[iblk], 
            z_spos[iblk], z_epos[iblk], 
            x_coef, y_coef, z_coef, x_out
        );
    }
}