#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "Poisson_FD3D_PBC_FFT_Solver.h"

double fract(const int n, const int k) 
{
    double Nr = 1.0, Dr = 1.0, val;
    for (int i = n-k+1; i <= n; i++) Nr *= i;
    for (int i = n+1; i <= n+k; i++) Dr *= i;
    val = Nr / Dr;
    return val;
}

double calcL2NormRelErr(const int n, const double *x0, const double *x1)
{
    double res, x0_l2 = 0.0, diff_l2 = 0.0, diff;
    for (int i = 0; i < n; i++)
    {
        diff     = x0[i] - x1[i];
        x0_l2   += x0[i] * x0[i];
        diff_l2 +=  diff *  diff;
    }
    res = sqrt(diff_l2) / sqrt(x0_l2);
    return res;
}

int main()
{
    // Finite difference domain
    const double Lx = 23.34;
    const double Ly = 23.34;
    const double Lz = 23.34;
    const int    nx = 36;
    const int    ny = 36;
    const int    nz = 36;
    const int    nd = nx * nx * nz;
    const double dx = Lx / (double) nx;
    const double dy = Ly / (double) ny;
    const double dz = Lz / (double) nz;
    
    // FD weights
    const int radius  = 1;
    const int radius1 = radius + 1;
    double *FD_coef   = (double*) malloc(sizeof(double) * 4 * radius1);
    double *w2_coef   = FD_coef;
    double *w2_x_coef = FD_coef + radius1;
    double *w2_y_coef = FD_coef + radius1 * 2;
    double *w2_z_coef = FD_coef + radius1 * 3;
    w2_coef[0] = 0.0;
    for (int r = 1; r < radius1; r++) 
    {
        w2_coef[0] -= (2.0 / (r*r));
        w2_coef[r] = (2 * (r%2) - 1) * 2 * fract(radius, r) / (r*r);  
    }
    for (int r = 0; r < radius1; r++) 
    {
        w2_x_coef[r] = w2_coef[r] / (dx * dx);
        w2_y_coef[r] = w2_coef[r] / (dy * dy);
        w2_z_coef[r] = w2_coef[r] / (dz * dz);
    }
    
    // Read RHS and reference solution from file
    FILE *f_rhs_inf = fopen("f_rhs.txt", "r");
    FILE *u_ref_inf = fopen("u_ref.txt", "r");
    double *f_rhs   = (double*) malloc(sizeof(double) * nd);
    double *u_ref   = (double*) malloc(sizeof(double) * nd);
    double *u_fft   = (double*) malloc(sizeof(double) * nd);
    assert(f_rhs != NULL && u_ref != NULL && u_fft != NULL);
    for (int i = 0; i < nd; i++) 
    {
        fscanf(f_rhs_inf, "%lf", f_rhs + i);
        fscanf(u_ref_inf, "%lf", u_ref + i);
    }
    fclose(f_rhs_inf);
    fclose(u_ref_inf);
    
    // Solve Poisson equation with PBC using FFT
    printf("FFT warm-up running:\n");
    Poisson_FD3D_PBC_FFT_Solver(nx, ny, nz, radius, w2_x_coef, w2_y_coef, w2_z_coef, f_rhs, u_fft);
    printf("Performance test running:\n");
    Poisson_FD3D_PBC_FFT_Solver(nx, ny, nz, radius, w2_x_coef, w2_y_coef, w2_z_coef, f_rhs, u_fft);
    
    // Compare the results
    double relerr = calcL2NormRelErr(nd, u_ref, u_fft);
    printf("||u_{ref} - u_{fft}||_2 / ||u_{ref}||_2 = %e\n", relerr);
    
    free(u_fft);
    free(u_ref);
    free(f_rhs);
    free(FD_coef);
}