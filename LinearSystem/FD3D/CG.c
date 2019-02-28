#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "FD3D.h"

// Classic Conjugate Gradients method for solving Laplacian * x = b
void CG_classic(
    const int n, const double res_tol, const int max_iter,
    const double *b, double *x
)
{
    size_t buff_vec_msize = sizeof(double) * n;
    double *CG_buff = (double*) malloc(buff_vec_msize * 3);
    assert(CG_buff != NULL);
    
    double *r = CG_buff + n * 0;
    double *p = CG_buff + n * 1;
    double *s = CG_buff + n * 2;
    double r2 = 0.0, r2_old = 0.0;
    
    double st = omp_get_wtime();
    
    FD3D_Laplacian_MatVec(x, r);
    #pragma omp parallel for simd reduction(+:r2, r2_old)
    for (int i = 0; i < n; i++)
    {
        r[i]   = b[i] - r[i];   // r = b - A * x;
        p[i]   = r[i];          // p = r;
        r2     += r[i] * r[i];  // rho1 = r' * r;
        r2_old += b[i] * b[i];  // b_l2norm = norm(b, 2);
    }
    
    double b_l2norm = sqrt(r2_old);
    double residual = sqrt(r2);
    double res_stop = b_l2norm * res_tol;
    int    iter_cnt = 1;
    printf("Iteration, Relative Residual\n");
    printf("      %3d,         %e\n", 0, residual / b_l2norm);
    
    double tmp0, tmp1, alpha, beta, omega;
    for (; iter_cnt < max_iter; iter_cnt++)
    {
        // (1) s = A * p;
        FD3D_Laplacian_MatVec(p, s);
        
        // (2) alpha = r2 / (p' * s);
        tmp0 = 0.0;
        #pragma omp parallel for simd reduction(+:tmp0)
        for (int i = 0; i < n; i++)
            tmp0 += p[i] * s[i];
        alpha = r2 / tmp0;
        
        // (3) x = x + alpha * p;
        // (4) r = r - alpha * s;
        #pragma omp parallel for simd reduction(+:tmp0)
        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * s[i];
        }
        
        // (5) r2_old = r2; r2 = r' * r;
        r2_old = r2;
        r2 = 0.0;
        #pragma omp parallel for simd reduction(+:r2)
        for (int i = 0; i < n; i++)
            r2 += r[i] * r[i];
        
        // (6) beta = r2 / r2_old; p = r + beta * p;
        beta = r2 / r2_old;
        #pragma omp parallel for simd reduction(+:r2)
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * p[i];
        
        residual = sqrt(r2);
        
        printf("      %3d,         %e\n", iter_cnt, residual / b_l2norm);
        if (residual < res_stop) 
        {
            double et = omp_get_wtime();
            printf("Classic CG converged, used time = %lf (s)\n", et - st);
            break;
        }
    }
    
    free(CG_buff);
}
