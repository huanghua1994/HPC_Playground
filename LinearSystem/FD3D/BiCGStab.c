#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "FD3D.h"

// BiConjugate Gradients Stabilized method for solving Laplacian * x = b
void BiCGStab(
    const int n, const double res_tol, const int max_iter,
    const double *b, double *x
)
{
    size_t buff_vec_msize = sizeof(double) * n;
    double *BiCGStab_buff = (double*) malloc(buff_vec_msize * 6);
    assert(BiCGStab_buff != NULL);
    
    double *r   = BiCGStab_buff + n * 0;
    double *p   = BiCGStab_buff + n * 1;
    double *s   = BiCGStab_buff + n * 2;
    double *Ap  = BiCGStab_buff + n * 3;
    double *As  = BiCGStab_buff + n * 4;
    double *r0s = BiCGStab_buff + n * 5;
    double rho0 = 0.0, rho1 = 0.0;
    
    double st = omp_get_wtime();
    
    FD3D_Laplacian_MatVec(x, r);
    #pragma omp parallel for simd reduction(+:rho0, rho1)
    for (int i = 0; i < n; i++)
    {
        r[i]   = b[i] - r[i];  // r = b - A * x;
        p[i]   = r[i];         // p = r;
        r0s[i] = r[i];         // r0s = r;
        rho1  += r[i] * r[i];  // rho1 = r' * r0s;
        rho0  += b[i] * b[i];  // b_l2norm = norm(b, 2);
    }
    
    double b_l2norm = sqrt(rho0);
    double residual = sqrt(rho1);
    double res_stop = b_l2norm * res_tol;
    int    iter_cnt = 1;

    // Ap = A * p;
    FD3D_Laplacian_MatVec(p, Ap);

    printf("Iteration, Relative Residual\n");
    printf("      %3d,         %e\n", 0, residual / b_l2norm);
    
    double tmp0, tmp1, alpha, beta, omega;
    for (; iter_cnt < max_iter; iter_cnt++)
    {
        // (1) alpha = r' * r0s / (Ap' * r0s);
        tmp0 = 0.0; tmp1 = 0.0;
        #pragma omp parallel for simd reduction(+:tmp0, tmp1)
        for (int i = 0; i < n; i++) 
        {
            tmp0 +=  r[i] * r0s[i];
            tmp1 += Ap[i] * r0s[i];
        }
        alpha = tmp0 / tmp1;
        
        // (2) s = r - alpha * Ap;
        #pragma omp parallel for simd
        for (int i = 0; i < n; i++) 
            s[i] = r[i] - alpha * Ap[i];
        
        // (3) As = A * s;
        FD3D_Laplacian_MatVec(s, As);
        
        // (4) omega = (As' * s) / (As' * As);
        tmp0 = 0.0; tmp1 = 0.0;
        #pragma omp parallel for simd reduction(+:tmp0, tmp1)
        for (int i = 0; i < n; i++) 
        {
            tmp0 += As[i] *  s[i];
            tmp1 += As[i] * As[i];
        }
        omega = tmp0 / tmp1;
        
        // (5) x = x + alpha * p + omega * s;
        #pragma omp parallel for simd
        for (int i = 0; i < n; i++)
            x[i] = x[i] + alpha * p[i] + omega * s[i];
        
        // (6) rho0 = rho1; r = s - omega * As; rho1 = r' * r0s;
        rho0 = rho1;
        rho1 = 0.0;
        #pragma omp parallel for simd reduction(+:rho1)
        for (int i = 0; i < n; i++)
        {
            r[i]  = s[i] - omega * As[i];
            rho1 += r[i] * r0s[i];
        }
        
        // (7) beta = rho1 / rho0 * alpha / omega;
        beta = rho1 / rho0 * alpha / omega;
        
        // (8) p = r + beta * (p - omega * Ap);
        #pragma omp parallel for simd
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * (p[i] - omega * Ap[i]);
        
        // (9) Ap = A * p;
        FD3D_Laplacian_MatVec(p, Ap);
        
        // (10) residual = norm(r, 2);
        residual = 0.0;
        #pragma omp parallel for simd reduction(+:residual)
        for (int i = 0; i < n; i++)
            residual += r[i] * r[i];
        residual = sqrt(residual);
        
        printf("      %3d,         %e\n", iter_cnt, residual / b_l2norm);
        if (residual < res_stop) 
        {
            double et = omp_get_wtime();
            printf("BiCGStab converged, used time = %lf (s)\n", et - st);
            break;
        }
    }
    
    free(BiCGStab_buff);
}
