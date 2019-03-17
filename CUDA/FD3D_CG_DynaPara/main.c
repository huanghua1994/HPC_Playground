#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include "CG.h"
#include "FD3D.h"
#include "FD3D_CG_cuda.h"

const int    RADIUS   = 6;
const int    max_iter = 500;
const double res_tol  = 1e-6;

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

int main(int argc, char **argv)
{
    int nx = 128, ny = 128, nz = 128;
    if (argc >= 2) nx = atoi(argv[1]);
    if (argc >= 3) nx = atoi(argv[2]);
    if (argc >= 4) nx = atoi(argv[3]);
    if (nx < 64 || nx > 1024) nx = 128;
    if (ny < 64 || ny > 1024) ny = 128;
    if (nz < 64 || nz > 1024) nz = 128;
    printf("3D finite difference domain = %d * %d * %d, order-12 stencil\n", nx, ny, nz);
    printf("Iterative solver max iteration = %d, residual tolerance = %e\n", max_iter, res_tol);
    
    int nxyz = nx * ny * nz;
    size_t nxyz_msize = sizeof(double) * nxyz;
    double *b  = (double*) malloc(nxyz_msize);
    double *x0 = (double*) malloc(nxyz_msize);
    double *x1 = (double*) malloc(nxyz_msize);
    assert(b != NULL && x0 != NULL && x1 != NULL);
    
    srand(114514);
    for (int i = 0; i < nxyz; i++)
    {
        b[i]  = (double) (i % 114);
        x0[i] = 0.0;
        x1[i] = 0.0;
    }
    FD3D_Laplacian_set_param(nx, ny, nz, RADIUS);
    
    printf("Starting CPU CG...\n");
    CG_classic(nxyz, res_tol, max_iter, b, x0);
    
    printf("Starting GPU CG...\n");
    FD3D_CG_cuda(nx, ny, nz, res_tol, max_iter, b, stencil_coefs, x1);
    
    double res = calcL2NormRelErr(nxyz, x0, x1);
    printf("||x_CPU - x_GPU||_2 / ||x_CPU||_2 = %e\n", res);
    
    free(b);
    free(x0);
    free(x1);
    return 0;
}