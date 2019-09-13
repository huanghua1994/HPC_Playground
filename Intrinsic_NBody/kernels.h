#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "x86_intrin_wrapper.h" 
#include <math.h>

#define DTYPE    double
#define DSQRT    sqrt
#define VEC_T    vec_t_d
#define SIMD_LEN SIMD_LEN_D

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

static void reciprocal_matvec_avx_aligned(
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
    for (int i = 0; i < n0; i += SIMD_LEN)
    {
        vec_t_d tx = intrin_load_d(x0 + i);
        vec_t_d ty = intrin_load_d(y0 + i);
        vec_t_d tz = intrin_load_d(z0 + i);
        vec_t_d tv = intrin_zero_d();
        for (int j = 0; j < n1; j++)
        {
            vec_t_d dx = intrin_sub_d(tx, intrin_bcast_d(x1 + j));
            vec_t_d dy = intrin_sub_d(ty, intrin_bcast_d(y1 + j));
            vec_t_d dz = intrin_sub_d(tz, intrin_bcast_d(z1 + j));
            
            vec_t_d r2 = intrin_mul_d(dx, dx);
            r2 = intrin_fmadd_d(dy, dy, r2);
            r2 = intrin_fmadd_d(dz, dz, r2);
            
            vec_t_d sv   = intrin_bcast_d(x_in + j);
            vec_t_d rinv = intrin_div_d(sv, intrin_sqrt_d(r2));
            tv = intrin_add_d(rinv, tv);
        }
        intrin_store_d(x_out + i, tv);
    }
}

#endif
