#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "Poisson_multigrid.h"
#include "CG.h"

int main(int argc, char **argv)
{
    double cell_dims[3] = {19.2, 19.2, 19.6};
    int grid_sizes[3] = {48, 48, 50};  // Nx, Ny, Nz
    int BCs[3] = {0, 0, 1};
    int FDn = 6;
    
    if (argc < 7)
    {
        printf("Nx, Ny, Nz: ");
        scanf("%d%d%d", &grid_sizes[0], &grid_sizes[1], &grid_sizes[2]);
        printf("BCx, BCy, BCz: ");
        scanf("%d%d%d", &BCs[0], &BCs[1], &BCs[2]);
    } else {
        grid_sizes[0] = atoi(argv[1]);
        grid_sizes[1] = atoi(argv[2]);
        grid_sizes[2] = atoi(argv[3]);
        BCs[0] = atoi(argv[4]);
        BCs[1] = atoi(argv[5]);
        BCs[2] = atoi(argv[6]);
    }
    cell_dims[0] = 0.4 * (double) (grid_sizes[0] - BCs[0]);
    cell_dims[1] = 0.4 * (double) (grid_sizes[1] - BCs[1]);
    cell_dims[2] = 0.4 * (double) (grid_sizes[2] - BCs[2]);
    
    printf("Problem setting: \n");
    printf("    Nx, Ny, Nz: %d, %d, %d\n", grid_sizes[0], grid_sizes[1], grid_sizes[2]);
    printf("    BCx, BCy, BCz: %d, %d, %d\n", BCs[0], BCs[1], BCs[2]);
    
    int Nd = grid_sizes[0] * grid_sizes[1] * grid_sizes[2];
    double *b = (double *) malloc(sizeof(double) * Nd);
    double *x = (double *) malloc(sizeof(double) * Nd);
    assert(b != NULL && x != NULL);
    
    srand48(19241112);
    for (int i = 0; i < Nd; i++) b[i] = drand48() - 0.5;
    
    // If all periodic boundaries, then project b such that the 
    // system is consistent
    if (BCs[0] == 0 && BCs[1] == 0 && BCs[2] == 0)
    {
        double t = 0.0;
        for (int i = 0; i < Nd; i++) t += b[i];
        t /= (double)(Nd);
        for (int i = 0; i < Nd; i++) b[i] -= t;
    }
    
    mg_data_t mg_data;
    MG_init(cell_dims, grid_sizes, BCs, FDn, &mg_data);
    MG_solve(mg_data, b, x, 1e-10);
    printf("%lf  %lf  %lf  %lf\n", x[0], x[1], x[2], x[3]);
    
    memset(x, 0, sizeof(double) * Nd);
    CG_classic(mg_data->mkl_sp_A[0], Nd, 1e-10, 2000, b, x);
    printf("%lf  %lf  %lf  %lf\n", x[0], x[1], x[2], x[3]);
    
    MG_destroy(mg_data);
    free(b);
    free(x);
    
    return 0;
}