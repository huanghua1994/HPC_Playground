#ifndef __RPY_KERNEL_H__
#define __RPY_KERNEL_H__

#include <math.h>
#include "x86_intrin_wrapper.h" 
#include "kernel_ptr.h"

// ===== Input & output vector are not transposed (npoint * 3 matrices) =====

static void RPY_matvec_nt_std(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, double *x_out_0
)
{
    const double a = 0.2, eta = 1.0;
    const double C   = 1.0 / (6.0 * M_PI * a * eta);
    const double a2  = 2.0 * a;
    const double aa2 = a * a * 2.0;
    const double aa_2o3   = aa2 / 3.0;
    const double C_075    = C * 0.75;
    const double C_9o32oa = C * 9.0 / 32.0 / a;
    const double C_3o32oa = C * 3.0 / 32.0 / a;
    for (int i = 0; i < n0; i++)
    {
        double tx = coord0[i];
        double ty = coord0[i + ld0];
        double tz = coord0[i + ld0 * 2];
        double res[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < n1; j++)
        {
            double dx = tx - coord1[j];
            double dy = ty - coord1[j + ld1];
            double dz = tz - coord1[j + ld1 * 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double r  = sqrt(r2);
            double inv_r = (r < 1e-15) ? 0.0 : 1.0 / r;
            
            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;

            double t1, t2;
            if (r < a2)
            {
                t1 = C - C_9o32oa * r;
                t2 =     C_3o32oa * r;
            } else {
                t1 = C_075 * inv_r * (1 + aa_2o3 / r2);
                t2 = C_075 * inv_r * (1 - aa2    / r2); 
            }
            
            double x_in_0_j[3];
            x_in_0_j[0] = x_in_0[j * 3 + 0];
            x_in_0_j[1] = x_in_0[j * 3 + 1];
            x_in_0_j[2] = x_in_0[j * 3 + 2];
            
            res[0] += (t2 * dx * dx + t1) * x_in_0_j[0];
            res[0] += (t2 * dx * dy)      * x_in_0_j[1];
            res[0] += (t2 * dx * dz)      * x_in_0_j[2];
            res[1] += (t2 * dy * dx)      * x_in_0_j[0];
            res[1] += (t2 * dy * dy + t1) * x_in_0_j[1];
            res[1] += (t2 * dy * dz)      * x_in_0_j[2];
            res[2] += (t2 * dz * dx)      * x_in_0_j[0];
            res[2] += (t2 * dz * dy)      * x_in_0_j[1];
            res[2] += (t2 * dz * dz + t1) * x_in_0_j[2];
        }
        x_out_0[i * 3 + 0] += res[0];
        x_out_0[i * 3 + 1] += res[1];
        x_out_0[i * 3 + 2] += res[2];
    }
}

static void RPY_matvec_nt_t_std(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    const double a = 0.2, eta = 1.0;
    const double C   = 1.0 / (6.0 * M_PI * a * eta);
    const double a2  = 2.0 * a;
    const double aa2 = a * a * 2.0;
    const double aa_2o3   = aa2 / 3.0;
    const double C_075    = C * 0.75;
    const double C_9o32oa = C * 9.0 / 32.0 / a;
    const double C_3o32oa = C * 3.0 / 32.0 / a;
    
    for (int i = 0; i < n0; i++)
    {
        double tx = coord0[i];
        double ty = coord0[i + ld0];
        double tz = coord0[i + ld0 * 2];
        double x_in_1_i[3];
        int i3 = i * 3;
        x_in_1_i[0] = x_in_1[i3 + 0];
        x_in_1_i[1] = x_in_1[i3 + 1];
        x_in_1_i[2] = x_in_1[i3 + 2];
        double res[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < n1; j++)
        {
            double dx = tx - coord1[j];
            double dy = ty - coord1[j + ld1];
            double dz = tz - coord1[j + ld1 * 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double r  = sqrt(r2);
            double inv_r = (r < 1e-15) ? 0.0 : 1.0 / r;
            
            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;

            double t1, t2;
            if (r < a2)
            {
                t1 = C - C_9o32oa * r;
                t2 =     C_3o32oa * r;
            } else {
                t1 = C_075 * inv_r * (1 + aa_2o3 / r2);
                t2 = C_075 * inv_r * (1 - aa2    / r2); 
            }
            
            double x_in_0_j[3];
            x_in_0_j[0] = x_in_0[j * 3 + 0];
            x_in_0_j[1] = x_in_0[j * 3 + 1];
            x_in_0_j[2] = x_in_0[j * 3 + 2];
            
            res[0] += (t2 * dx * dx + t1) * x_in_0_j[0];
            res[0] += (t2 * dx * dy)      * x_in_0_j[1];
            res[0] += (t2 * dx * dz)      * x_in_0_j[2];
            res[1] += (t2 * dy * dx)      * x_in_0_j[0];
            res[1] += (t2 * dy * dy + t1) * x_in_0_j[1];
            res[1] += (t2 * dy * dz)      * x_in_0_j[2];
            res[2] += (t2 * dz * dx)      * x_in_0_j[0];
            res[2] += (t2 * dz * dy)      * x_in_0_j[1];
            res[2] += (t2 * dz * dz + t1) * x_in_0_j[2];
            
            int j3 = j * 3;
            x_out_1[j3 + 0] += (t2 * dx * dx + t1) * x_in_1_i[0];
            x_out_1[j3 + 0] += (t2 * dy * dx)      * x_in_1_i[1];
            x_out_1[j3 + 0] += (t2 * dz * dx)      * x_in_1_i[2];
            x_out_1[j3 + 1] += (t2 * dx * dy)      * x_in_1_i[0];
            x_out_1[j3 + 1] += (t2 * dy * dy + t1) * x_in_1_i[1];
            x_out_1[j3 + 1] += (t2 * dz * dy)      * x_in_1_i[2];
            x_out_1[j3 + 2] += (t2 * dx * dz)      * x_in_1_i[0];
            x_out_1[j3 + 2] += (t2 * dy * dz)      * x_in_1_i[1];
            x_out_1[j3 + 2] += (t2 * dz * dz + t1) * x_in_1_i[2];
        }
        x_out_0[i * 3 + 0] += res[0];
        x_out_0[i * 3 + 1] += res[1];
        x_out_0[i * 3 + 2] += res[2];
    }
}

static void RPY_symm_matvec_std(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    if (x_in_1 == NULL)
    {
        RPY_matvec_nt_std(coord0, ld0, n0, coord1, ld1, n1, x_in_0, x_out_0);
    } else {
        RPY_matvec_nt_t_std(coord0, ld0, n0, coord1, ld1, n1, x_in_0, x_in_1, x_out_0, x_out_1);
    }
}

// ===== Input & output vector are transposed (3 * npoint matrices) =====

static void RPY_matvec_nt_autovec(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, double *x_out_0
)
{
    const double a = 0.2, eta = 1.0;
    const double C   = 1.0 / (6.0 * M_PI * a * eta);
    const double a2  = 2.0 * a;
    const double aa2 = a * a * 2.0;
    const double aa_2o3   = aa2 / 3.0;
    const double C_075    = C * 0.75;
    const double C_9o32oa = C * 9.0 / 32.0 / a;
    const double C_3o32oa = C * 3.0 / 32.0 / a;
    
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;

    for (int i = 0; i < n0; i++)
    {
        double txs = x0[i];
        double tys = y0[i];
        double tzs = z0[i];
        double res[3] = {0, 0, 0};
        for (int j = 0; j < n1; j++)
        {
            double dx = txs - x1[j];
            double dy = tys - y1[j];
            double dz = tzs - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double r  = sqrt(r2);
            double inv_r = (r < 1e-15) ? 0.0 : 1.0 / r;
            
            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;

            double t1, t2;
            if (r < a2)
            {
                t1 = C - C_9o32oa * r;
                t2 =     C_3o32oa * r;
            } else {
                t1 = C_075 * inv_r * (1 + aa_2o3 / r2);
                t2 = C_075 * inv_r * (1 - aa2    / r2); 
            }
            
            double x_in_0_j[3];
            x_in_0_j[0] = x_in_0[j + 0 * ld1];
            x_in_0_j[1] = x_in_0[j + 1 * ld1];
            x_in_0_j[2] = x_in_0[j + 2 * ld1];
            
            res[0] += (t2 * dx * dx + t1) * x_in_0_j[0];
            res[0] += (t2 * dx * dy)      * x_in_0_j[1];
            res[0] += (t2 * dx * dz)      * x_in_0_j[2];
            res[1] += (t2 * dy * dx)      * x_in_0_j[0];
            res[1] += (t2 * dy * dy + t1) * x_in_0_j[1];
            res[1] += (t2 * dy * dz)      * x_in_0_j[2];
            res[2] += (t2 * dz * dx)      * x_in_0_j[0];
            res[2] += (t2 * dz * dy)      * x_in_0_j[1];
            res[2] += (t2 * dz * dz + t1) * x_in_0_j[2];
        }
        x_out_0[i + 0 * ld0] += res[0];
        x_out_0[i + 1 * ld0] += res[1];
        x_out_0[i + 2 * ld0] += res[2];
    }
}

static void RPY_matvec_nt_t_autovec(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    const double a = 0.2, eta = 1.0;
    const double C   = 1.0 / (6.0 * M_PI * a * eta);
    const double a2  = 2.0 * a;
    const double aa2 = a * a * 2.0;
    const double aa_2o3   = aa2 / 3.0;
    const double C_075    = C * 0.75;
    const double C_9o32oa = C * 9.0 / 32.0 / a;
    const double C_3o32oa = C * 3.0 / 32.0 / a;
    
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;

    for (int i = 0; i < n0; i++)
    {
        double txs = x0[i];
        double tys = y0[i];
        double tzs = z0[i];
        double x_in_1_i[3];
        x_in_1_i[0] = x_in_1[i + 0 * ld0];
        x_in_1_i[1] = x_in_1[i + 1 * ld0];
        x_in_1_i[2] = x_in_1[i + 2 * ld0];
        double res[3] = {0, 0, 0};
        // Don't know why ICC cannot vectorize this loop correctly
        //#pragma omp simd  
        for (int j = 0; j < n1; j++)
        {
            double dx = txs - x1[j];
            double dy = tys - y1[j];
            double dz = tzs - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double r  = sqrt(r2);
            double inv_r = (r < 1e-15) ? 0.0 : 1.0 / r;
            
            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;

            double t1, t2;
            if (r < a2)
            {
                t1 = C - C_9o32oa * r;
                t2 =     C_3o32oa * r;
            } else {
                t1 = C_075 * inv_r * (1 + aa_2o3 / r2);
                t2 = C_075 * inv_r * (1 - aa2    / r2); 
            }
            
            double x_in_0_j[3];
            x_in_0_j[0] = x_in_0[j + 0 * ld1];
            x_in_0_j[1] = x_in_0[j + 1 * ld1];
            x_in_0_j[2] = x_in_0[j + 2 * ld1];
            
            double tmp0 = 0.0, tmp1 = 0.0, tmp2 = 0.0;
            double tmp, k0, k1, k2;
            tmp = t2 * dx;
            k0 = (tmp * dx + t1);
            k1 = (tmp * dy);
            k2 = (tmp * dz);
            res[0] += k0 * x_in_0_j[0];
            res[0] += k1 * x_in_0_j[1];
            res[0] += k2 * x_in_0_j[2];
            tmp0 += k0 * x_in_1_i[0];
            tmp1 += k1 * x_in_1_i[0];
            tmp2 += k2 * x_in_1_i[0];
            
            tmp = t2 * dy;
            k0 = (tmp * dx);
            k1 = (tmp * dy + t1);
            k2 = (tmp * dz);
            res[1] += k0 * x_in_0_j[0];
            res[1] += k1 * x_in_0_j[1];
            res[1] += k2 * x_in_0_j[2];
            tmp0 += k0 * x_in_1_i[1];
            tmp1 += k1 * x_in_1_i[1];
            tmp2 += k2 * x_in_1_i[1];
            
            tmp = t2 * dz;
            k0 = (tmp * dx);
            k1 = (tmp * dy);
            k2 = (tmp * dz + t1);
            res[2] += k0 * x_in_0_j[0];
            res[2] += k1 * x_in_0_j[1];
            res[2] += k2 * x_in_0_j[2];
            tmp0 += k0 * x_in_1_i[2];
            tmp1 += k1 * x_in_1_i[2];
            tmp2 += k2 * x_in_1_i[2];
            
            x_out_1[j + 0 * ld1] += tmp0;
            x_out_1[j + 1 * ld1] += tmp1;
            x_out_1[j + 2 * ld1] += tmp2;
        }
        x_out_0[i + 0 * ld0] += res[0];
        x_out_0[i + 1 * ld0] += res[1];
        x_out_0[i + 2 * ld0] += res[2];
    }
}


static void RPY_symm_matvec_autovec(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    if (x_in_1 == NULL)
    {
        RPY_matvec_nt_autovec(coord0, ld0, n0, coord1, ld1, n1, x_in_0, x_out_0);
    } else {
        //RPY_matvec_nt_t_autovec(coord0, ld0, n0, coord1, ld1, n1, x_in_0, x_in_1, x_out_0, x_out_1);
        // The following is actually faster since they can be auto vectorized
        RPY_matvec_nt_autovec(coord0, ld0, n0, coord1, ld1, n1, x_in_0, x_out_0);
        RPY_matvec_nt_autovec(coord1, ld1, n1, coord0, ld0, n0, x_in_1, x_out_1);
    }
}

// ===== Input & output vector are transposed (3 * npoint matrices) + intrinsic =====

static void RPY_matvec_nt_intrin(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, double *x_out_0
)
{
    const double a = 0.2, eta = 1.0;
    const double C   = 1.0 / (6.0 * M_PI * a * eta);
    const double a2  = 2.0 * a;
    const double aa2 = a * a * 2.0;
    const double aa_2o3   = aa2 / 3.0;
    const double C_075    = C * 0.75;
    const double C_9o32oa = C * 9.0 / 32.0 / a;
    const double C_3o32oa = C * 3.0 / 32.0 / a;
    
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;

    for (int i = 0; i < n0; i++)
    {
        int n1_SIMD = n1 / SIMD_LEN_D * SIMD_LEN_D;
        vec_d txv = vec_bcast_d(x0 + i);
        vec_d tyv = vec_bcast_d(y0 + i);
        vec_d tzv = vec_bcast_d(z0 + i);
        vec_d res0 = vec_zero_d();
        vec_d res1 = vec_zero_d();
        vec_d res2 = vec_zero_d();
        vec_d frsqrt_pf = vec_frsqrt_pf_d();
        for (int j = 0; j < n1_SIMD; j += SIMD_LEN_D)
        {
            vec_d dx = vec_sub_d(txv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(tyv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(tzv, vec_loadu_d(z1 + j));
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            vec_d r = vec_sqrt_d(r2);
            vec_d inv_r = vec_mul_d(vec_frsqrt_d(r2), frsqrt_pf);
            
            dx = vec_mul_d(dx, inv_r);
            dy = vec_mul_d(dy, inv_r);
            dz = vec_mul_d(dz, inv_r);
            
            vec_cmp_d r_lt_a2 = vec_cmp_lt_d(r, vec_set1_d(a2));
            vec_d C_075_o_r = vec_mul_d(vec_set1_d(C_075), inv_r);
            vec_d inv_r2 = vec_mul_d(inv_r, inv_r);
            
            vec_d tmp0, tmp1, t1, t2;
            tmp0 = vec_fnmadd_d(vec_set1_d(C_9o32oa), r, vec_set1_d(C));
            tmp1 = vec_fmadd_d(vec_set1_d(aa_2o3), inv_r2, vec_set1_d(1));
            tmp1 = vec_mul_d(C_075_o_r, tmp1);
            t1   = vec_blend_d(tmp1, tmp0, r_lt_a2);
            
            tmp0 = vec_mul_d(vec_set1_d(C_3o32oa), r);
            tmp1 = vec_fnmadd_d(vec_set1_d(aa2), inv_r2, vec_set1_d(1));
            tmp1 = vec_mul_d(C_075_o_r, tmp1);
            t2   = vec_blend_d(tmp1, tmp0, r_lt_a2);
            
            vec_d x_in_0_j0 = vec_loadu_d(x_in_0 + j);
            vec_d x_in_0_j1 = vec_loadu_d(x_in_0 + j + ld1);
            vec_d x_in_0_j2 = vec_loadu_d(x_in_0 + j + ld1 * 2);
            
            tmp0 = vec_mul_d(t2, dx);
            res0 = vec_fmadd_d(vec_fmadd_d(tmp0, dx, t1), x_in_0_j0, res0);
            res0 = vec_fmadd_d(vec_mul_d(tmp0, dy),       x_in_0_j1, res0);
            res0 = vec_fmadd_d(vec_mul_d(tmp0, dz),       x_in_0_j2, res0);
            tmp0 = vec_mul_d(t2, dy);
            res1 = vec_fmadd_d(vec_mul_d(tmp0, dx),       x_in_0_j0, res1);
            res1 = vec_fmadd_d(vec_fmadd_d(tmp0, dy, t1), x_in_0_j1, res1);
            res1 = vec_fmadd_d(vec_mul_d(tmp0, dz),       x_in_0_j2, res1);
            tmp0 = vec_mul_d(t2, dz);
            res2 = vec_fmadd_d(vec_mul_d(tmp0, dx),       x_in_0_j0, res2);
            res2 = vec_fmadd_d(vec_mul_d(tmp0, dy),       x_in_0_j1, res2);
            res2 = vec_fmadd_d(vec_fmadd_d(tmp0, dz, t1), x_in_0_j2, res2);
        }
        
        x_out_0[i + 0 * ld0] += vec_reduce_add_d(res0);
        x_out_0[i + 1 * ld0] += vec_reduce_add_d(res1);
        x_out_0[i + 2 * ld0] += vec_reduce_add_d(res2);
        RPY_matvec_nt_autovec(
            coord0 + i, ld0, 1, 
            coord1 + n1_SIMD, ld1, n1 - n1_SIMD, 
            x_in_0 + n1_SIMD, x_out_0 + i
        );
    }
}

static void RPY_matvec_nt_t_intrin(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    const double a = 0.2, eta = 1.0;
    const double C   = 1.0 / (6.0 * M_PI * a * eta);
    const double a2  = 2.0 * a;
    const double aa2 = a * a * 2.0;
    const double aa_2o3   = aa2 / 3.0;
    const double C_075    = C * 0.75;
    const double C_9o32oa = C * 9.0 / 32.0 / a;
    const double C_3o32oa = C * 3.0 / 32.0 / a;
    
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;

    for (int i = 0; i < n0; i++)
    {
        int n1_SIMD = n1 / SIMD_LEN_D * SIMD_LEN_D;
        vec_d txv = vec_bcast_d(x0 + i);
        vec_d tyv = vec_bcast_d(y0 + i);
        vec_d tzv = vec_bcast_d(z0 + i);
        vec_d x_in_1_i0 = vec_bcast_d(x_in_1 + i + 0 * ld0);
        vec_d x_in_1_i1 = vec_bcast_d(x_in_1 + i + 1 * ld0);
        vec_d x_in_1_i2 = vec_bcast_d(x_in_1 + i + 2 * ld0);
        vec_d res0 = vec_zero_d();
        vec_d res1 = vec_zero_d();
        vec_d res2 = vec_zero_d();
        vec_d frsqrt_pf = vec_frsqrt_pf_d();
        for (int j = 0; j < n1_SIMD; j += SIMD_LEN_D)
        {
            vec_d dx = vec_sub_d(txv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(tyv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(tzv, vec_loadu_d(z1 + j));
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            vec_d r = vec_sqrt_d(r2);
            vec_d inv_r = vec_mul_d(vec_frsqrt_d(r2), frsqrt_pf);
            
            dx = vec_mul_d(dx, inv_r);
            dy = vec_mul_d(dy, inv_r);
            dz = vec_mul_d(dz, inv_r);
            
            vec_cmp_d r_lt_a2 = vec_cmp_lt_d(r, vec_set1_d(a2));
            vec_d C_075_o_r = vec_mul_d(vec_set1_d(C_075), inv_r);
            vec_d inv_r2 = vec_mul_d(inv_r, inv_r);
            
            vec_d tmp0, tmp1, tmp2, t1, t2;
            tmp0 = vec_fnmadd_d(vec_set1_d(C_9o32oa), r, vec_set1_d(C));
            tmp1 = vec_fmadd_d(vec_set1_d(aa_2o3), inv_r2, vec_set1_d(1));
            tmp1 = vec_mul_d(C_075_o_r, tmp1);
            t1   = vec_blend_d(tmp1, tmp0, r_lt_a2);
            
            tmp0 = vec_mul_d(vec_set1_d(C_3o32oa), r);
            tmp1 = vec_fnmadd_d(vec_set1_d(aa2), inv_r2, vec_set1_d(1));
            tmp1 = vec_mul_d(C_075_o_r, tmp1);
            t2   = vec_blend_d(tmp1, tmp0, r_lt_a2);
            
            vec_d x_in_0_j0 = vec_loadu_d(x_in_0 + j);
            vec_d x_in_0_j1 = vec_loadu_d(x_in_0 + j + ld1);
            vec_d x_in_0_j2 = vec_loadu_d(x_in_0 + j + ld1 * 2);
            
            vec_d k0, k1, k2, tmp;
            tmp0 = vec_zero_d();
            tmp1 = vec_zero_d();
            tmp2 = vec_zero_d();
            
            tmp  = vec_mul_d(t2, dx);
            k0   = vec_fmadd_d(tmp, dx, t1);
            k1   = vec_mul_d(tmp, dy);
            k2   = vec_mul_d(tmp, dz);
            res0 = vec_fmadd_d(k0, x_in_0_j0, res0);
            res0 = vec_fmadd_d(k1, x_in_0_j1, res0);
            res0 = vec_fmadd_d(k2, x_in_0_j2, res0);
            tmp0 = vec_fmadd_d(k0, x_in_1_i0, tmp0);
            tmp1 = vec_fmadd_d(k1, x_in_1_i0, tmp1);
            tmp2 = vec_fmadd_d(k2, x_in_1_i0, tmp2);
            
            tmp  = vec_mul_d(t2, dy);
            k0   = vec_mul_d(tmp, dx);
            k1   = vec_fmadd_d(tmp, dy, t1);
            k2   = vec_mul_d(tmp, dz);
            res1 = vec_fmadd_d(k0, x_in_0_j0, res1);
            res1 = vec_fmadd_d(k1, x_in_0_j1, res1);
            res1 = vec_fmadd_d(k2, x_in_0_j2, res1);
            tmp0 = vec_fmadd_d(k0, x_in_1_i1, tmp0);
            tmp1 = vec_fmadd_d(k1, x_in_1_i1, tmp1);
            tmp2 = vec_fmadd_d(k2, x_in_1_i1, tmp2);
            
            tmp  = vec_mul_d(t2, dz);
            k0   = vec_mul_d(tmp, dx);
            k1   = vec_mul_d(tmp, dy);
            k2   = vec_fmadd_d(tmp, dz, t1);
            res2 = vec_fmadd_d(k0, x_in_0_j0, res2);
            res2 = vec_fmadd_d(k1, x_in_0_j1, res2);
            res2 = vec_fmadd_d(k2, x_in_0_j2, res2);
            tmp0 = vec_fmadd_d(k0, x_in_1_i2, tmp0);
            tmp1 = vec_fmadd_d(k1, x_in_1_i2, tmp1);
            tmp2 = vec_fmadd_d(k2, x_in_1_i2, tmp2);
            
            double *x_out_1_0 = x_out_1 + j + 0 * ld1;
            double *x_out_1_1 = x_out_1 + j + 1 * ld1;
            double *x_out_1_2 = x_out_1 + j + 2 * ld1;
            vec_storeu_d(x_out_1_0, vec_add_d(tmp0, vec_loadu_d(x_out_1_0)));
            vec_storeu_d(x_out_1_1, vec_add_d(tmp1, vec_loadu_d(x_out_1_1)));
            vec_storeu_d(x_out_1_2, vec_add_d(tmp2, vec_loadu_d(x_out_1_2)));
        }
        x_out_0[i + 0 * ld0] += vec_reduce_add_d(res0);
        x_out_0[i + 1 * ld0] += vec_reduce_add_d(res1);
        x_out_0[i + 2 * ld0] += vec_reduce_add_d(res2);
        
        RPY_symm_matvec_autovec(
            coord0 + i, ld0, 1, 
            coord1 + n1_SIMD, ld1, n1 - n1_SIMD, 
            x_in_0 + n1_SIMD, x_in_1 + i, 
            x_out_0 + i, x_out_1 + n1_SIMD
        );
    }
}

static void RPY_symm_matvec_intrin(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    if (x_in_1 == NULL)
    {
        RPY_matvec_nt_intrin(coord0, ld0, n0, coord1, ld1, n1, x_in_0, x_out_0);
    } else {
        RPY_matvec_nt_t_intrin(coord0, ld0, n0, coord1, ld1, n1, x_in_0, x_in_1, x_out_0, x_out_1);
    }
}


#endif
