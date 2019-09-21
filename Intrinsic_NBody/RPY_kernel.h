#ifndef __RPY_KERNEL_H__
#define __RPY_KERNEL_H__

#include <math.h>
#include "x86_intrin_wrapper.h" 
#include "kernel_ptr.h"

static void RPY_matvec_std(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, double *x_out_0
)
{
    const double a = 1.0, eta = 1.0;
    const double C   = 1.0 / (6.0 * M_PI * a * eta);
    const double a2  = 2.0 * a;
    const double aa2 = a * a * 2.0;
    const double aa_2o3   = aa2 / 3.0;
    const double C_075    = C * 0.75;
    const double C_9o32oa = C * 9.0 / 32.0 / a;
    const double C_3o32oa = C * 3.0 / 32.0 / a;

    for (int i = 0; i < n0; i++)
    {
        double x0 = coord0[i];
        double y0 = coord0[i + ld0];
        double z0 = coord0[i + ld0 * 2];
        double res[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < n1; j++)
        {
            double r0 = x0 - coord1[j];
            double r1 = y0 - coord1[j + ld1];
            double r2 = z0 - coord1[j + ld1 * 2];
            double s2 = r0 * r0 + r1 * r1 + r2 * r2;
            double s  = sqrt(s2);
            double x_in_0_j[3];
            x_in_0_j[0] = x_in_0[j * 3 + 0];
            x_in_0_j[1] = x_in_0[j * 3 + 1];
            x_in_0_j[2] = x_in_0[j * 3 + 2];

            if (s < 1e-15)
            {
                res[0] += C * x_in_0_j[0];
                res[1] += C * x_in_0_j[1];
                res[2] += C * x_in_0_j[2];
                continue;
            }
            
            double inv_s = 1.0 / s;
            r0 *= inv_s;
            r1 *= inv_s;
            r2 *= inv_s;
            double t1, t2;
            if (s < a2)
            {
                t1 = C - C_9o32oa * s;
                t2 =     C_3o32oa * s;
            } else {
                t1 = C_075 / s * (1 + aa_2o3 / s2);
                t2 = C_075 / s * (1 - aa2    / s2); 
            }

            res[0] += (t2 * r0 * r0 + t1) * x_in_0_j[0];
            res[0] += (t2 * r0 * r1)      * x_in_0_j[1];
            res[0] += (t2 * r0 * r2)      * x_in_0_j[2];
            res[1] += (t2 * r1 * r0)      * x_in_0_j[0];
            res[1] += (t2 * r1 * r1 + t1) * x_in_0_j[1];
            res[1] += (t2 * r1 * r2)      * x_in_0_j[2];
            res[2] += (t2 * r2 * r0)      * x_in_0_j[0];
            res[2] += (t2 * r2 * r1)      * x_in_0_j[1];
            res[2] += (t2 * r2 * r2 + t1) * x_in_0_j[2];
        }
        x_out_0[i * 3 + 0] += res[0];
        x_out_0[i * 3 + 1] += res[1];
        x_out_0[i * 3 + 2] += res[2];
    }
}

#endif
