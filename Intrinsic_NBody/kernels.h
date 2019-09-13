#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "x86_intrin_wrapper.h" 

// Pointer to function that performs kernel matrix matvec using given sets of 
// points and given input vector. The kernel function must be symmetric.
// Input parameters:
//   coord0 : Matrix, size 3-by-ld0, coordinates of the 1st point set
//   ld0    : Leading dimension of coord0, should be >= n0
//   n0     : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1 : Matrix, size 3-by-ld1, coordinates of the 2nd point set
//   ld1    : Leading dimension of coord1, should be >= n1
//   n1     : Number of points in coord1 (each column in coord0 is a coordinate)
//   x_in   : Vector, size >= n1, will be left multiplied by kernel_matrix(coord0, coord1)
// Output parameter:
//   x_out : Vector, size >= n0, x_out += kernel_matrix(coord0, coord1) * x_in
typedef void (*kernel_matvec_fptr) (
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const DTYPE *x_in, DTYPE *x_out
);

static void reciprocal_matvec_ref(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const DTYPE *x_in, DTYPE *x_out
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *z0 = coord0 + ld0 * 2;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    const DTYPE *z1 = coord1 + ld1 * 2;
    for (int i = 0; i < n0; i++)
    {
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        DTYPE sum = 0.0;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-20) r2 = 1.0;
            sum += x_in[j] / DSQRT(r2);
        }
        x_out[i] += sum;
    }
}

#endif
