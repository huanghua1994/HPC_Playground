#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "Poisson_multigrid.h"

int main(int argc, char **argv)
{
    double cell_dims[3] = {19.2, 19.2, 19.6};
    int grid_sizes[3] = {48, 48, 50};  // Nx, Ny, Nz
    int BCs[3] = {0, 0, 1};
    int FDn = 6;
    
    double st, et;
    
    mg_data_t mg_data;
    st = omp_get_wtime();
    MG_init(cell_dims, grid_sizes, BCs, FDn, &mg_data);
    et = omp_get_wtime();
    printf("MG_init() used %.3lf (s)\n", et - st);
    
    int Nd = grid_sizes[0] * grid_sizes[1] * grid_sizes[2];
    double *b = (double *) malloc(sizeof(double) * Nd);
    double *x = (double *) malloc(sizeof(double) * Nd);
    assert(b != NULL && x != NULL);
    
    srand48(19241112);
    for (int i = 0; i < Nd; i++) b[i] = drand48() - 0.5;
    
    st = omp_get_wtime();
    MG_solve(mg_data, b, x, 1e-9);
    et = omp_get_wtime();
    printf("MG_solve() used %.3lf (s)\n", et - st);
    
    MG_destroy(mg_data);
    
    return 0;
}