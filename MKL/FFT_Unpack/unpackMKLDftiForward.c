// See: https://software.intel.com/sites/default/files/managed/42/5b/Unpack_result_of_Intel_Math_Kernel_Library_Fast_Fourier_Transform_routines_to_align_with_Matlab.pdf

#include <complex.h>

#include "unpackMKLDftiForward.h"

// Unpack MKL 1D FFT CCE output format after 2D FFT forward transform
void unpackMKLDftiForward1DInplace(const int nx, double _Complex *data)
{
    int semi_nx = nx / 2 + 1;
    for (int dst_x = semi_nx; dst_x < nx; dst_x++)
    {
        int src_x = nx - dst_x;
        data[dst_x] = conj(data[src_x]);
    }
}

// Unpack MKL 2D FFT CCE output format after 2D FFT forward transform
void unpackMKLDftiForward2DInplace(const int ny, const int nx, double _Complex *data)
{
    int semi_nx = nx / 2 + 1;
    for (int dst_y = 0; dst_y < ny; dst_y++)
    {
        int src_y = (dst_y == 0) ? 0 : (ny - dst_y);
        int src_offset = src_y * nx;
        int dst_offset = dst_y * nx;
        for (int dst_x = semi_nx; dst_x < nx; dst_x++)
        {
            int src_x = nx - dst_x;
            data[dst_offset + dst_x] = conj(data[src_offset + src_x]);
        }
    }
}

// Unpack MKL 3D FFT CCE output format after 3D FFT forward transform
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
