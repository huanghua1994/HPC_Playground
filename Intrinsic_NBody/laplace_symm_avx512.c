#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "laplace_kernel.h"

void test_direct_nbody_symm(
    const int n_src, const double *src_coord, double *src_val0, double *src_val1, 
    const int n_trg, const double *trg_coord, double *trg_val0, double *trg_val1, 
    kernel_symm_matvec_fptr krnl_matvec, const int krnl_dim, int krnl_flops
)
{
    int nthread = omp_get_max_threads();
    
    double krnl_GFLOPS = (double)n_src * (double)n_trg * (double)krnl_flops;
    krnl_GFLOPS /= 1000000000.0;
    
    for (int k = 0; k < 5; k++)
    {
        memset(trg_val0, 0, sizeof(double) * n_trg * krnl_dim);
        memset(trg_val1, 0, sizeof(double) * n_src * krnl_dim);
        double st = omp_get_wtime();
        
        krnl_matvec(
            trg_coord, n_trg, n_trg, 
            src_coord, n_src, n_src, 
            src_val0, src_val1, trg_val0, trg_val1
        );
        double et = omp_get_wtime();
        double ut = et - st;
        printf("Direct N-Body %2d: %.3lf s, %.2lf GFLOPS\n", k, ut, krnl_GFLOPS / ut);
    }
}

static void laplace_3d_matvec_nt_t_std(
    const double *coord0, const int ld0, const int n0, 
    const double *coord1, const int ld1, const int n1, 
    const double *x_in_0, const double *x_in_1,         
    double *x_out_0, double *x_out_1
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
        const double x_in_1_i = x_in_1[i];
        double sum = 0.0;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            double dx = x0_i - x1[j];
            double dy = y0_i - y1[j];
            double dz = z0_i - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv_d = (r2 == 0.0) ? 0.0 : (1.0 / sqrt(r2));
            sum += x_in_0[j] * inv_d;
            x_out_1[j] += x_in_1_i * inv_d;
        }
        x_out_0[i] += sum;
    }
}

#ifdef USE_AVX512

// ======================= From H2Pack_kernels.h ======================= //
#define SYMM_MATVEC_KRNL_PARAM \
    const double *coord0, const int ld0, const int n0, \
    const double *coord1, const int ld1, const int n1, \
    const double *x_in_0, const double *x_in_1,         \
    double *x_out_0, double *x_out_1

#define EXTRACT_3D_COORD() \
    const double *x0 = coord0 + ld0 * 0; \
    const double *y0 = coord0 + ld0 * 1; \
    const double *z0 = coord0 + ld0 * 2; \
    const double *x1 = coord1 + ld1 * 0; \
    const double *y1 = coord1 + ld1 * 1; \
    const double *z1 = coord1 + ld1 * 2; 

#define SIMD_LEN 8
static void Coulomb_3d_matvec_nt_t_intrin_d(SYMM_MATVEC_KRNL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n0_vec  = (n0 / SIMD_LEN) * SIMD_LEN;
    const int n1_vec  = (n1 / SIMD_LEN) * SIMD_LEN;
    const int n0_vec2 = (n0 / 2) * 2;
    const vec_d frsqrt_pf = vec_frsqrt_pf_d();
    // 2-way unroll to reduce update of x_out_1
    for (int i = 0; i < n0_vec2; i += 2)
    {
        vec_d sum_v0 = vec_zero_d();
        vec_d sum_v1 = vec_zero_d();
        vec_d x0_i0v = vec_bcast_d(x0 + i);
        vec_d y0_i0v = vec_bcast_d(y0 + i);
        vec_d z0_i0v = vec_bcast_d(z0 + i);
        vec_d x0_i1v = vec_bcast_d(x0 + i + 1);
        vec_d y0_i1v = vec_bcast_d(y0 + i + 1);
        vec_d z0_i1v = vec_bcast_d(z0 + i + 1);
        vec_d x_in_1_i0v = vec_bcast_d(x_in_1 + i);
        vec_d x_in_1_i1v = vec_bcast_d(x_in_1 + i + 1);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d dx = vec_sub_d(x0_i0v, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(y0_i0v, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(z0_i0v, vec_loadu_d(z1 + j));
            
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            vec_d rinv0 = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r2));
            
            
            dx = vec_sub_d(x0_i1v, vec_loadu_d(x1 + j));
            dy = vec_sub_d(y0_i1v, vec_loadu_d(y1 + j));
            dz = vec_sub_d(z0_i1v, vec_loadu_d(z1 + j));
            
            r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            vec_d rinv1 = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r2));
            
            vec_d x_in_0_j = vec_loadu_d(x_in_0 + j);
            sum_v0 = vec_fmadd_d(x_in_0_j, rinv0, sum_v0);
            sum_v1 = vec_fmadd_d(x_in_0_j, rinv1, sum_v1);
            
            vec_d ov1 = vec_loadu_d(x_out_1 + j);
            ov1 = vec_fmadd_d(x_in_1_i0v, rinv0, ov1);
            ov1 = vec_fmadd_d(x_in_1_i1v, rinv1, ov1);
            vec_storeu_d(x_out_1 + j, ov1);
        }
        
        const double x0_i0 = x0[i];
        const double y0_i0 = y0[i];
        const double z0_i0 = z0[i];
        const double x0_i1 = x0[i + 1];
        const double y0_i1 = y0[i + 1];
        const double z0_i1 = z0[i + 1];
        const double x_in_1_i0 = x_in_1[i];
        const double x_in_1_i1 = x_in_1[i + 1];
        double sum0 = vec_reduce_add_d(sum_v0);
        double sum1 = vec_reduce_add_d(sum_v1);
        for (int j = n1_vec; j < n1; j++)
        {
            double dx = x0_i0 - x1[j];
            double dy = y0_i0 - y1[j];
            double dz = z0_i0 - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv_d0 = (r2 == 0.0) ? 0.0 : (1.0 / sqrt(r2));
            
            dx = x0_i1 - x1[j];
            dy = y0_i1 - y1[j];
            dz = z0_i1 - z1[j];
            r2 = dx * dx + dy * dy + dz * dz;
            double inv_d1 = (r2 == 0.0) ? 0.0 : (1.0 / sqrt(r2));
            
            sum0 += x_in_0[j] * inv_d0;
            sum1 += x_in_0[j] * inv_d1;
            x_out_1[j] += x_in_1_i0 * inv_d0;
            x_out_1[j] += x_in_1_i1 * inv_d1;
        }
        x_out_0[i]     += sum0;
        x_out_0[i + 1] += sum1;
    }
    
    for (int i = n0_vec2; i < n0; i++)
    {
        vec_d x0_iv = vec_bcast_d(x0 + i);
        vec_d y0_iv = vec_bcast_d(y0 + i);
        vec_d z0_iv = vec_bcast_d(z0 + i);
        vec_d x_in_1_iv = vec_bcast_d(x_in_1 + i);
        vec_d sum_v = vec_zero_d();
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d dx = vec_sub_d(x0_iv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(y0_iv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(z0_iv, vec_loadu_d(z1 + j));
            
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            
            vec_d rinv = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r2));
            sum_v = vec_fmadd_d(vec_loadu_d(x_in_0 + j), rinv, sum_v);
            
            vec_d ov1 = vec_loadu_d(x_out_1 + j);
            ov1 = vec_fmadd_d(x_in_1_iv, rinv, ov1);
            vec_storeu_d(x_out_1 + j, ov1);
        }
        
        const double x0_i = x0[i];
        const double y0_i = y0[i];
        const double z0_i = z0[i];
        const double x_in_1_i = x_in_1[i];
        double sum = vec_reduce_add_d(sum_v);
        for (int j = n1_vec; j < n1; j++)
        {
            double dx = x0_i - x1[j];
            double dy = y0_i - y1[j];
            double dz = z0_i - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv_d = (r2 == 0.0) ? 0.0 : (1.0 / sqrt(r2));
            sum += x_in_0[j] * inv_d;
            x_out_1[j] += x_in_1_i * inv_d;
        }
        x_out_0[i] += sum;
    }
}
// ===================================================================== //

static inline void print_m512d(__m512d x)
{
    double *p = (double*) &x;
    printf("%e %e %e %e %e %e %e %e\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}

static inline void laplace_symm_matvec_8x8d(
    __m512d tx, __m512d ty, __m512d tz, 
    __m512d sx, __m512d sy, __m512d sz, 
    __m512d sv0, __m512d sv1, __m512d *tv0, __m512d *tv1
)
{
    __m512d dx, dy, dz, r2, res0, res1, tmp1;
    __m512i swap_idx;
    
    res0 = vec_zero_d();
    res1 = vec_zero_d();
    
    // (0) [x0, x1, x2, x3, x4, x5, x6, x7]
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    res1 = vec_mul_d(sv1, r2);
    
    // (1) [x1, x0, x3, x2, x5, x4, x7, x6]
    // 0x55 = 0b01010101
    sx  = _mm512_shuffle_pd(sx,  sx,  0x55);
    sy  = _mm512_shuffle_pd(sy,  sy,  0x55);
    sz  = _mm512_shuffle_pd(sz,  sz,  0x55);
    sv0 = _mm512_shuffle_pd(sv0, sv0, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    tmp1 = _mm512_shuffle_pd(tmp1, tmp1, 0x55);
    res1 = vec_add_d(tmp1, res1);
    
    // (2) [x3, x2, x1, x0, x7, x6, x5, x4]
    // 0xB1 = 0b10110001
    sx  = _mm512_shuffle_f64x2(sx,  sx,  0xB1);
    sy  = _mm512_shuffle_f64x2(sy,  sy,  0xB1);
    sz  = _mm512_shuffle_f64x2(sz,  sz,  0xB1);
    sv0 = _mm512_shuffle_f64x2(sv0, sv0, 0xB1);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    tmp1 = _mm512_shuffle_pd(tmp1, tmp1, 0x55);
    tmp1 = _mm512_shuffle_f64x2(tmp1, tmp1, 0xB1);
    res1 = vec_add_d(tmp1, res1);
    
    // (3) [x2, x3, x0, x1, x6, x7, x4, x5]
    // 0x55 = 0b01010101
    sx  = _mm512_shuffle_pd(sx,  sx,  0x55);
    sy  = _mm512_shuffle_pd(sy,  sy,  0x55);
    sz  = _mm512_shuffle_pd(sz,  sz,  0x55);
    sv0 = _mm512_shuffle_pd(sv0, sv0, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    tmp1 = _mm512_shuffle_f64x2(tmp1, tmp1, 0xB1);
    res1 = vec_add_d(tmp1, res1);
    
    // (4) [x6, x7, x4, x5, x2, x3, x0, x1]
    swap_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    sx  = _mm512_permutexvar_pd(swap_idx, sx);
    sy  = _mm512_permutexvar_pd(swap_idx, sy);
    sz  = _mm512_permutexvar_pd(swap_idx, sz);
    sv0 = _mm512_permutexvar_pd(swap_idx, sv0);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    swap_idx = _mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6);
    tmp1 = _mm512_permutexvar_pd(swap_idx, tmp1);
    res1 = vec_add_d(tmp1, res1);
    
    // (5) [x7, x6, x5, x4, x3, x2, x1, x0]
    // 0x55 = 0b01010101
    sx  = _mm512_shuffle_pd(sx,  sx,  0x55);
    sy  = _mm512_shuffle_pd(sy,  sy,  0x55);
    sz  = _mm512_shuffle_pd(sz,  sz,  0x55);
    sv0 = _mm512_shuffle_pd(sv0, sv0, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    swap_idx = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    tmp1 = _mm512_permutexvar_pd(swap_idx, tmp1);
    res1 = vec_add_d(tmp1, res1);
    
    // (6) [x5, x4, x7, x6, x1, x0, x3, x2]
    // 0xB1 = 0b10110001
    sx  = _mm512_shuffle_f64x2(sx,  sx,  0xB1);
    sy  = _mm512_shuffle_f64x2(sy,  sy,  0xB1);
    sz  = _mm512_shuffle_f64x2(sz,  sz,  0xB1);
    sv0 = _mm512_shuffle_f64x2(sv0, sv0, 0xB1);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    swap_idx = _mm512_set_epi64(2, 3, 0, 1, 6, 7, 4, 5);
    tmp1 = _mm512_permutexvar_pd(swap_idx, tmp1);
    res1 = vec_add_d(tmp1, res1);
    
    // (7) [x4, x5, x6, x7, x0, x1, x2, x3]
    // 0x55 = 0b01010101
    sx  = _mm512_shuffle_pd(sx,  sx,  0x55);
    sy  = _mm512_shuffle_pd(sy,  sy,  0x55);
    sz  = _mm512_shuffle_pd(sz,  sz,  0x55);
    sv0 = _mm512_shuffle_pd(sv0, sv0, 0x55);
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    swap_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    tmp1 = _mm512_permutexvar_pd(swap_idx, tmp1);
    res1 = vec_add_d(tmp1, res1);
    
    *tv0 = res0;
    *tv1 = res1;
}

#endif  // End of #ifdef USE_AVX512

static void laplace_symm_matvec_avx_new(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1,         
    double *x_out_0, double *x_out_1
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
            vec_d tx  = vec_loadu_d(x0 + i);
            vec_d ty  = vec_loadu_d(y0 + i);
            vec_d tz  = vec_loadu_d(z0 + i);
            vec_d tv  = vec_zero_d();
            vec_d sv1 = vec_loadu_d(x_in_1 + i);
            for (int j = j_sidx; j < j_eidx; j += SIMD_LEN_D)
            {
                vec_d sx  = vec_loadu_d(x1 + j);
                vec_d sy  = vec_loadu_d(y1 + j);
                vec_d sz  = vec_loadu_d(z1 + j);
                vec_d sv0 = vec_loadu_d(x_in_0 + j);
                
                vec_d tmp0, tmp1;
                laplace_symm_matvec_8x8d(tx, ty, tz, sx, sy, sz, sv0, sv1, &tmp0, &tmp1);
                
                tv = vec_add_d(tmp0, tv);
                tmp1 = vec_mul_d(tmp1, frsqrt_pf);
                tmp1 = vec_add_d(vec_loadu_d(x_out_1 + j), tmp1);
                vec_storeu_d(x_out_1 + j, tmp1);
            }
            tv = vec_mul_d(tv, frsqrt_pf);
            vec_d ov0 = vec_loadu_d(x_out_0 + i);
            vec_storeu_d(x_out_0 + i, vec_add_d(ov0, tv));
        }
    }
    laplace_3d_matvec_nt_t_std(
        coord0, ld0, n0_SIMD,
        coord1 + n1_SIMD, ld1, n1 - n1_SIMD,
        x_in_0 + n1_SIMD, x_in_1, x_out_0, x_out_1 + n1_SIMD
    );
    laplace_3d_matvec_nt_t_std(
        coord0 + n0_SIMD, ld0, n0 - n0_SIMD,
        coord1, ld1, n1,
        x_in_0, x_in_1 + n0_SIMD, x_out_0 + n0_SIMD, x_out_1
    );
}

int main(int argc, char **argv)
{
    int n_src = 0, n_trg = 0;
    if (argc >= 2) n_src = atoi(argv[1]);
    if (argc >= 3) n_trg = atoi(argv[2]);
    if (n_src == 0)
    {
        printf("n_src = ");
        scanf("%d", &n_src);
    }
    if (n_trg == 0)
    {
        printf("n_trg = ");
        scanf("%d", &n_trg);
    }
    
    int krnl_dim = 1;
    
    double *src_coord = (double*) malloc(sizeof(double) * n_src * 3);
    double *trg_coord = (double*) malloc(sizeof(double) * n_trg * 3);
    double *src_val0  = (double*) malloc(sizeof(double) * n_src * krnl_dim);
    double *src_val1  = (double*) malloc(sizeof(double) * n_trg * krnl_dim);
    double *trg_val0  = (double*) malloc(sizeof(double) * n_trg * krnl_dim);
    double *trg_val1  = (double*) malloc(sizeof(double) * n_src * krnl_dim);
    double *trg_val2  = (double*) malloc(sizeof(double) * n_trg * krnl_dim);
    double *trg_val3  = (double*) malloc(sizeof(double) * n_src * krnl_dim);
    srand48(114);
    for (int i = 0; i < n_src; i++) 
    {
        src_coord[i + n_src * 0] = drand48();
        src_coord[i + n_src * 1] = drand48();
        src_coord[i + n_src * 2] = drand48();
    }
    for (int i = 0; i < n_src * krnl_dim; i++) 
    {
        src_val0[i] = drand48();
        trg_val1[i] = 0;
        trg_val3[i] = 0;
    }
    for (int i = 0; i < n_trg * krnl_dim; i++) 
    {
        src_val1[i] = drand48();
        trg_val0[i] = 0;
        trg_val2[i] = 0;
    }
    for (int i = 0; i < n_trg; i++) 
    {
        trg_coord[i + n_trg * 0] = drand48();
        trg_coord[i + n_trg * 1] = drand48();
        trg_coord[i + n_trg * 2] = drand48();
    }
    
    
    printf("laplace_3d_matvec_nt_t_std: \n");
    test_direct_nbody_symm(
        n_src, src_coord, src_val0, src_val1,
        n_trg, trg_coord, trg_val0, trg_val1,
        laplace_3d_matvec_nt_t_std, 1, 14
    );
    
    printf("laplace_symm_matvec_avx_new: \n");
    test_direct_nbody_symm(
        n_src, src_coord, src_val0, src_val1,
        n_trg, trg_coord, trg_val2, trg_val3,
        laplace_symm_matvec_avx_new, 1, 22
    );
    
    double ref_l2_0 = 0.0, err_l2_0 = 0.0;
    double ref_l2_1 = 0.0, err_l2_1 = 0.0;
    for (int i = 0; i < n_trg * krnl_dim; i++)
    {
        double diff = trg_val0[i] - trg_val2[i];
        ref_l2_0 += trg_val0[i] * trg_val0[i];
        err_l2_0 += diff * diff;
    }
    for (int i = 0; i < n_src * krnl_dim; i++)
    {
        double diff = trg_val1[i] - trg_val3[i];
        ref_l2_1 += trg_val1[i] * trg_val1[i];
        err_l2_1 += diff * diff;
    }
    ref_l2_0 = sqrt(ref_l2_0);
    err_l2_0 = sqrt(err_l2_0);
    ref_l2_1 = sqrt(ref_l2_1);
    err_l2_1 = sqrt(err_l2_1);
    printf("relative L2 error = %e, %e\n", err_l2_0 / ref_l2_0, err_l2_1 / ref_l2_1);
    
    return 0;
}