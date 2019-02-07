#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>


void stencil_3axis_ref_thread(
    const double *x0, const int radius,
    const int nx, const int ny, const int nz, 
    const int x_spos, const int x_epos, 
    const int y_spos, const int y_epos,
    const int z_spos, const int z_epos,
    const double *stencil_coefs, double *x1
)
{
    const int nx_ex = nx + 2 * radius;
    const int ny_ex = ny + 2 * radius;
    const int stride_y = nx;
    const int stride_z = nx * ny;
    const int stride_y_ex = nx_ex;
    const int stride_z_ex = nx_ex * ny_ex;
    const double coef_0 = stencil_coefs[0] * stencil_coefs[0] * stencil_coefs[0];

    for (int z = z_spos; z < z_epos; z++)
    {
        int iz = z + radius;
        for (int y = y_spos; y < y_epos; y++)
        {
            int iy = y + radius;
            int offset = z * stride_z + y * stride_y;
            int offset_ex = iz * stride_z_ex + iy * stride_y_ex;
            for (int x = x_spos; x < x_epos; x++)
            {
                int idx_ex = offset_ex + x + radius;
                double res = coef_0 * x0[idx_ex];
                for (int r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double tmp, c = stencil_coefs[r];
                    tmp  = x0[idx_ex + r] + x0[idx_ex - r];
                    tmp += x0[idx_ex + stride_y_r] + x0[idx_ex - stride_y_r];
                    tmp += x0[idx_ex + stride_z_r] + x0[idx_ex - stride_z_r];
                    res += c * tmp;
                }
                x1[offset + x] = res;
            }
        }
    }
}

void stencil_3axis_ref(
    const double *x0, const int radius, const double *stencil_coefs, 
    const int nx, const int ny, const int nz, double *x1
)
{
    #pragma omp parallel for
    for (int iz = 0; iz < nz; iz++)
    {
        stencil_3axis_ref_thread(
            x0, radius, nx, ny, nz, 0, nx, 0, ny, 
            iz, iz + 1, stencil_coefs, x1
        );
    }
}