#ifndef __RECIPROCAL_KERNELS_H__
#define __RECIPROCAL_KERNELS_H__

#include <math.h>
#include <arm_sve.h>

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
            double res = x_in[j] / sqrt(r2); //(r2 == 0.0) ? 0.0 : (x_in[j] / sqrt(r2));
            sum += res;
        }
        x_out[i] += sum;
    }
}

#define RSQRT_REFINE(pg, rsqrt_target, rsqrt_iter, rsqrt_work)  \
    do  \
    {   \
        rsqrt_work = svmul_f64_z(pg, rsqrt_target, rsqrt_iter); \
        rsqrt_work = svrsqrts_f64(rsqrt_work, rsqrt_iter);      \
        rsqrt_iter = svmul_f64_z(pg, rsqrt_work, rsqrt_iter);   \
    } while (0)
    

static void reciprocal_matvec_intrin(
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
    const int blk_size   = 1024;
    const int SIMD_LEN_D = svcntd();
    svbool_t ptrue64b = svptrue_b64();
    for (int j_start = 0; j_start < n1; j_start += blk_size)
    {
        int j_end = (j_start + blk_size > n1) ? n1 : (j_start + blk_size);
        int i = 0;
        svbool_t pg = svwhilelt_b64(i, n0);
        do
        {
            svfloat64_t vec_tx = svld1(pg, x0 + i);
            svfloat64_t vec_ty = svld1(pg, y0 + i);
            svfloat64_t vec_tz = svld1(pg, z0 + i);
            svfloat64_t vec_tv = svld1(pg, x_out + i);
            for (int j = j_start; j < j_end; j++)
            {
                svfloat64_t vec_dx = svsub_f64_z(pg, vec_tx, svdup_f64_z(pg, x1[j]));
                svfloat64_t vec_dy = svsub_f64_z(pg, vec_ty, svdup_f64_z(pg, y1[j]));
                svfloat64_t vec_dz = svsub_f64_z(pg, vec_tz, svdup_f64_z(pg, z1[j]));
                svfloat64_t vec_sv = svdup_f64_z(pg, x_in[j]);

                svfloat64_t vec_r2 = svmul_f64_z(pg, vec_dx, vec_dx);
                vec_r2 = svmad_f64_z(pg, vec_dy, vec_dy, vec_r2);
                vec_r2 = svmad_f64_z(pg, vec_dz, vec_dz, vec_r2);

                svfloat64_t vec_rsqrt = svrsqrte_f64(vec_r2);
                svfloat64_t vec_rsqrt_work;
                RSQRT_REFINE(pg, vec_r2, vec_rsqrt, vec_rsqrt_work);
                RSQRT_REFINE(pg, vec_r2, vec_rsqrt, vec_rsqrt_work);
                vec_tv = svmad_f64_z(pg, vec_rsqrt, vec_sv, vec_tv);
            }
            svst1(pg, x_out + i, vec_tv);

            i += SIMD_LEN_D;
            pg = svwhilelt_b64(i, n0);
        } while (svptest_any(ptrue64b, pg));
    }
}

#endif
