#ifndef __RECIPROCAL_KERNELS_H__
#define __RECIPROCAL_KERNELS_H__

#include <math.h>
#include "avx_intrin_wrapper.h"
#include "kernel_ptr.h"

static void reciprocal_matvec_std(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
)
{
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    for (int i = 0; i < n0; i++)
    {
        const double x0_i = x0[i];
        const double y0_i = y0[i];
        const double z0_i = z0[i];
        double sum = 0.0;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            double dx = x0_i - x1[j];
            double dy = y0_i - y1[j];
            double dz = z0_i - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double res = (r2 == 0.0) ? 0.0 : (x_in[j] / sqrt(r2));
            sum += res;
        }
        x_out[i] += sum;
    }
}

static void reciprocal_matvec_avx(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
)
{
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    vec_d frsqrt_pf = vec_frsqrt_pf_d();
    int i;
    const int blk_size = 1024;
    for (int j_sidx = 0; j_sidx < n1; j_sidx += blk_size)
    {
        int j_eidx = (j_sidx + blk_size > n1) ? n1 : (j_sidx + blk_size);
        for (i = 0; i <= n0 - SIMD_LEN_D; i += SIMD_LEN_D)
        {
            vec_d tx = vec_loadu_d(x0 + i);
            vec_d ty = vec_loadu_d(y0 + i);
            vec_d tz = vec_loadu_d(z0 + i);
            vec_d tv = vec_zero_d();
            for (int j = j_sidx; j < j_eidx; j++)
            {
                vec_d dx = vec_sub_d(tx, vec_bcast_d(x1 + j));
                vec_d dy = vec_sub_d(ty, vec_bcast_d(y1 + j));
                vec_d dz = vec_sub_d(tz, vec_bcast_d(z1 + j));
                
                vec_d r2 = vec_mul_d(dx, dx);
                r2 = vec_fmadd_d(dy, dy, r2);
                r2 = vec_fmadd_d(dz, dz, r2);
                
                vec_d sv = vec_mul_d(vec_bcast_d(x_in + j), frsqrt_pf);
                vec_d rinv = vec_frsqrt_d(r2);
                tv = vec_fmadd_d(rinv, sv, tv);
            }
            vec_d outval = vec_loadu_d(x_out + i);
            vec_storeu_d(x_out + i, vec_add_d(outval, tv));
        }
    }
    reciprocal_matvec_std(
        coord0 + i, ld0, n0 - i,
        coord1, ld1, n1,
        x_in, x_out + i
    );
}

#ifdef USE_AVX
static inline __m256d reciprocal_matvec_4x4d(
    __m256d tx, __m256d ty, __m256d tz, 
    __m256d sx, __m256d sy, __m256d sz, __m256d sv
)
{
    __m256d dx, dy, dz, r2, res;
    
    res = vec_zero_d();
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    sx = _mm256_shuffle_pd(sx, sx, 0x5);
    sy = _mm256_shuffle_pd(sy, sy, 0x5);
    sz = _mm256_shuffle_pd(sz, sz, 0x5);
    sv = _mm256_shuffle_pd(sv, sv, 0x5);
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    sx = _mm256_permute2f128_pd(sx, sx, 0x1);
    sy = _mm256_permute2f128_pd(sy, sy, 0x1);
    sz = _mm256_permute2f128_pd(sz, sz, 0x1);
    sv = _mm256_permute2f128_pd(sv, sv, 0x1);
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    sx = _mm256_shuffle_pd(sx, sx, 0x5);
    sy = _mm256_shuffle_pd(sy, sy, 0x5);
    sz = _mm256_shuffle_pd(sz, sz, 0x5);
    sv = _mm256_shuffle_pd(sv, sv, 0x5);
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    return res;
}
#endif  // End of #ifdef USE_AVX

#ifdef USE_AVX512
static inline __m512d reciprocal_matvec_8x8d(
    __m512d tx, __m512d ty, __m512d tz, 
    __m512d sx, __m512d sy, __m512d sz, __m512d sv
)
{
    __m512d dx, dy, dz, r2, res;
    
    res = vec_zero_d();
    
    // (0) [x0, x1, x2, x3, x4, x5, x6, x7]
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    // (1) [x1, x0, x3, x2, x5, x4, x7, x6]
    // 0x55 = 0b01010101
    sx = _mm512_shuffle_pd(sx, sx, 0x55);
    sy = _mm512_shuffle_pd(sy, sy, 0x55);
    sz = _mm512_shuffle_pd(sz, sz, 0x55);
    sv = _mm512_shuffle_pd(sv, sv, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    // (2) [x3, x2, x1, x0, x7, x6, x5, x4]
    // 0xB1 = 0b10110001
    sx = _mm512_shuffle_f64x2(sx, sx, 0xB1);
    sy = _mm512_shuffle_f64x2(sy, sy, 0xB1);
    sz = _mm512_shuffle_f64x2(sz, sz, 0xB1);
    sv = _mm512_shuffle_f64x2(sv, sv, 0xB1);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    // (3) [x2, x3, x0, x1, x6, x7, x4, x5]
    // 0x55 = 0b01010101
    sx = _mm512_shuffle_pd(sx, sx, 0x55);
    sy = _mm512_shuffle_pd(sy, sy, 0x55);
    sz = _mm512_shuffle_pd(sz, sz, 0x55);
    sv = _mm512_shuffle_pd(sv, sv, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    // (4) [x6, x7, x4, x5, x2, x3, x0, x1]
    __m512i swap256 = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    sx = _mm512_permutexvar_pd(swap256, sx);
    sy = _mm512_permutexvar_pd(swap256, sy);
    sz = _mm512_permutexvar_pd(swap256, sz);
    sv = _mm512_permutexvar_pd(swap256, sv);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    // (5) [x7, x6, x5, x4, x3, x2, x1, x0]
    // 0x55 = 0b01010101
    sx = _mm512_shuffle_pd(sx, sx, 0x55);
    sy = _mm512_shuffle_pd(sy, sy, 0x55);
    sz = _mm512_shuffle_pd(sz, sz, 0x55);
    sv = _mm512_shuffle_pd(sv, sv, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    // (6) [x5, x4, x7, x6, x1, x0, x3, x2]
    // 0xB1 = 0b10110001
    sx = _mm512_shuffle_f64x2(sx, sx, 0xB1);
    sy = _mm512_shuffle_f64x2(sy, sy, 0xB1);
    sz = _mm512_shuffle_f64x2(sz, sz, 0xB1);
    sv = _mm512_shuffle_f64x2(sv, sv, 0xB1);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    // (7) [x4, x5, x6, x7, x0, x1, x2, x3]
    // 0x55 = 0b01010101
    sx = _mm512_shuffle_pd(sx, sx, 0x55);
    sy = _mm512_shuffle_pd(sy, sy, 0x55);
    sz = _mm512_shuffle_pd(sz, sz, 0x55);
    sv = _mm512_shuffle_pd(sv, sv, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res = vec_fmadd_d(sv, r2, res);
    
    return res;
}
#endif  // End of #ifdef USE_AVX512

static void reciprocal_matvec_avx_new(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
)
{
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    vec_d frsqrt_pf = vec_frsqrt_pf_d();
    const int blk_size = 1024;
    int n0_SIMD = (n0 / SIMD_LEN_D) * SIMD_LEN_D;
    int n1_SIMD = (n1 / SIMD_LEN_D) * SIMD_LEN_D;
    for (int j_sidx = 0; j_sidx < n1_SIMD; j_sidx += blk_size)
    {
        int j_eidx = (j_sidx + blk_size > n1_SIMD) ? n1_SIMD : (j_sidx + blk_size);
        for (int i = 0; i < n0_SIMD; i += SIMD_LEN_D)
        {
            vec_d tx = vec_loadu_d(x0 + i);
            vec_d ty = vec_loadu_d(y0 + i);
            vec_d tz = vec_loadu_d(z0 + i);
            vec_d tv = vec_zero_d();
            for (int j = j_sidx; j < j_eidx; j += SIMD_LEN_D)
            {
                vec_d sx = vec_loadu_d(x1 + j);
                vec_d sy = vec_loadu_d(y1 + j);
                vec_d sz = vec_loadu_d(z1 + j);
                vec_d sv = vec_loadu_d(x_in + j);
                
                #ifdef USE_AVX
                vec_d tmp = reciprocal_matvec_4x4d(tx, ty, tz, sx, sy, sz, sv);
                #endif
                #ifdef USE_AVX512
                vec_d tmp = reciprocal_matvec_8x8d(tx, ty, tz, sx, sy, sz, sv);
                #endif
                tv = vec_add_d(tmp, tv);
            }
            tv = vec_mul_d(tv, frsqrt_pf);
            vec_d outval = vec_loadu_d(x_out + i);
            vec_storeu_d(x_out + i, vec_add_d(outval, tv));
        }
    }
    reciprocal_matvec_std(
        coord0, ld0, n0_SIMD,
        coord1 + n1_SIMD, ld1, n1 - n1_SIMD,
        x_in + n1_SIMD, x_out
    );
    reciprocal_matvec_std(
        coord0 + n0_SIMD, ld0, n0 - n0_SIMD,
        coord1, ld1, n1,
        x_in, x_out + n0_SIMD
    );
}

#endif
