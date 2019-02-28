#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "AAR.h"
#include "BiCGStab.h"
#include "CG.h"
#include "FD3D.h"

const int    RADIUS   = 6;
const int    max_iter = 500;
const double res_tol  = 1e-6;

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
    double *b = (double*) malloc(nxyz_msize);
    double *x = (double*) malloc(nxyz_msize);
    assert(b != NULL && x != NULL);
    
    srand(time(NULL));
    for (int i = 0; i < nxyz; i++)
        b[i] = (double) rand() / (double) RAND_MAX;
    
    FD3D_Laplacian_set_param(nx, ny, nz, RADIUS);
    
    /*
    printf("Starting BiCGStab...\n");
    // Use x = 0 as initial guess
    memset(x, 0, nxyz_msize);
    BiCGStab(nxyz, res_tol, max_iter, b, x);
    
    printf("Starting classic CG...\n");
    // Use x = 0 as initial guess
    memset(x, 0, nxyz_msize);
    CG_classic(nxyz, res_tol, max_iter, b, x);
    */
    
    printf("Starting AAR...\n");
    // Use x = 0 as initial guess
    memset(x, 0, nxyz_msize);
    AAR(nxyz, res_tol, max_iter, b, x);
    
    free(b);
    free(x);
    return 0;
}