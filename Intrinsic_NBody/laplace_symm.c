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

#ifdef USE_AVX
static inline void print_m256d(__m256d x)
{
    double *p = (double*) &x;
    printf("%e %e %e %e\n", p[0], p[1], p[2], p[3]);
}

static inline void laplace_symm_matvec_4x4d(
    __m256d tx, __m256d ty, __m256d tz, 
    __m256d sx, __m256d sy, __m256d sz, 
    __m256d sv0, __m256d sv1, __m256d *tv0, __m256d *tv1
)
{
    __m256d dx, dy, dz, r2, res0, res1;
    
    res0 = vec_zero_d();
    res1 = vec_zero_d();
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    res1 = vec_fmadd_d(sv1, r2, res1);
    
    sx  = _mm256_shuffle_pd(sx,  sx,  0x5);
    sy  = _mm256_shuffle_pd(sy,  sy,  0x5);
    sz  = _mm256_shuffle_pd(sz,  sz,  0x5);
    sv0 = _mm256_shuffle_pd(sv0, sv0, 0x5);
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    vec_d tmp1 = vec_mul_d(sv1, r2);
    // tmp1 = [y0 y1 y2 y3], we need [y1 y0 y3 y2] here
    tmp1 = _mm256_shuffle_pd(tmp1, tmp1, 0x5);
    res1 = vec_add_d(tmp1, res1);
    
    sx  = _mm256_permute2f128_pd(sx,  sx,  0x1);
    sy  = _mm256_permute2f128_pd(sy,  sy,  0x1);
    sz  = _mm256_permute2f128_pd(sz,  sz,  0x1);
    sv0 = _mm256_permute2f128_pd(sv0, sv0, 0x1);
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    // tmp1 = [y0 y1 y2 y3], we need [y3 y2 y1 y0] here
    tmp1 = _mm256_shuffle_pd(tmp1, tmp1, 0x5);
    tmp1 = _mm256_permute2f128_pd(tmp1, tmp1, 0x1);
    res1 = vec_add_d(tmp1, res1);
    
    sx  = _mm256_shuffle_pd(sx,  sx,  0x5);
    sy  = _mm256_shuffle_pd(sy,  sy,  0x5);
    sz  = _mm256_shuffle_pd(sz,  sz,  0x5);
    sv0 = _mm256_shuffle_pd(sv0, sv0, 0x5);
    
    dx = vec_sub_d(tx, sx);
    dy = vec_sub_d(ty, sy);
    dz = vec_sub_d(tz, sz);
    r2 = vec_mul_d(dx, dx);
    r2 = vec_fmadd_d(dy, dy, r2);
    r2 = vec_fmadd_d(dz, dz, r2);
    r2 = vec_frsqrt_d(r2);
    res0 = vec_fmadd_d(sv0, r2, res0);
    tmp1 = vec_mul_d(sv1, r2);
    // tmp1 = [y0 y1 y2 y3], we need [y2 y3 y0 y1] here
    tmp1 = _mm256_permute2f128_pd(tmp1, tmp1, 0x1);
    res1 = vec_add_d(tmp1, res1);
    
    *tv0 = res0;
    *tv1 = res1;
}
#endif  // End of #ifdef USE_AVX

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
                laplace_symm_matvec_4x4d(tx, ty, tz, sx, sy, sz, sv0, sv1, &tmp0, &tmp1);
                
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