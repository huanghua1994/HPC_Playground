#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "Poisson_multigrid.h"

int main(int argc, char **argv)
{
    double cell_dims[3] = {20.0, 20.0, 30.0};
    int grid_sizes[3] = {48, 48, 50};  // Nx, Ny, Nz
    int BCs[3] = {0, 0, 1};
    int FDn = 6;
    
    mg_data_t mg_data;
    MG_init(cell_dims, grid_sizes, BCs, FDn, &mg_data);
    
    
    MG_destroy(mg_data);
    
    return 0;
}