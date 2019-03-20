#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "mmio.h"
#include "CSRPlus.h"
#include "magma_v2.h"
#include "magmasparse.h"

void read_mm_to_COO(
    FILE *inf, int *nrows_, int *ncols_, int *nnz_,
    int **row_, int **col_, double **val_
)
{
    MM_typecode mat_type;
    if (mm_read_banner(inf, &mat_type) != 0)
    {
        printf("Could not process Matrix Market banner\n");
        exit(1);
    }
    
    if (mm_is_complex(mat_type) || mm_is_pattern(mat_type))
    {
        printf("Do not support complex or pattern matrix\n");
        exit(1);
    }
    
    int issym = 0, isreal = 0;
    if (mm_is_symmetric(mat_type)) issym = 1;
    if (mm_is_real(mat_type)) isreal = 1;
    
    int nrows, ncols, nelem, nnz;
    mm_read_mtx_crd_size(inf, &nrows, &ncols, &nelem);
    nnz = nelem + issym * nelem;  // Actually we may not have so many non-zeros
    int    *row = (int*)    malloc(sizeof(int)    * nnz);
    int    *col = (int*)    malloc(sizeof(int)    * nnz);
    double *val = (double*) malloc(sizeof(double) * nnz);
    assert(row != NULL && col != NULL && val != NULL);
    nnz = 0;
    if (isreal == 1 && issym == 1)
    {
        for (int i = 0; i < nelem; i++)
        {
            fscanf(inf, "%d %d %lf\n", row + nnz, col + nnz, val + nnz);
            if (row[nnz] != col[nnz])
            {
                row[nnz + 1] = col[nnz];
                col[nnz + 1] = row[nnz];
                val[nnz + 1] = val[nnz];
                nnz += 2;
            } 
            else nnz++;
        }
    }
    if (isreal == 0 && issym == 1)
    {
        int ival;
        for (int i = 0; i < nelem; i++)
        {
            fscanf(inf, "%d %d %d\n", row + nnz, col + nnz, &ival);
            val[nnz] = (double) ival;
            if (row[nnz] != col[nnz])
            {
                row[nnz + 1] = col[nnz];
                col[nnz + 1] = row[nnz];
                val[nnz + 1] = val[nnz];
                nnz += 2;
            } 
            else nnz++;
        }
    }
    if (isreal == 1 && issym == 0)
    {
        for (int i = 0; i < nelem; i++)
            fscanf(inf, "%d %d %lf\n", row + i, col + i, val + i);
        nnz = nelem;
    }
    if (isreal == 0 && issym == 0)
    {
        int ival;
        for (int i = 0; i < nelem; i++)
        {
            fscanf(inf, "%d %d %d\n", row + i, col + i, &ival);
            val[i] = (double) ival;
        }
        nnz = nelem;
    }
    
    // Adjust 1-based index to 0-based index
    for (int i = 0; i < nnz; i++)
    {
        row[i]--;
        col[i]--;
    }
    
    *nrows_ = nrows;
    *ncols_ = ncols;
    *nnz_   = nnz;
    *row_   = row;
    *col_   = col;
    *val_   = val;
}

int main(int argc, char **argv)
{
    if (argc < 3) 
    {
        printf("Usage: %s <martix-market-filename> <ntest>\n", argv[0]);
        exit(1);
    }
    
    FILE *inf = fopen(argv[1], "r");
    int ntest = atoi(argv[2]);
    if (ntest < 1 || ntest > 100) ntest = 10;
    if (inf == NULL)
    {
        printf("Cannot open file: %s\n", argv[1]);
        exit(1);
    }
    
    int nrows, ncols, nnz;
    int *row, *col;
    double *val;
    read_mm_to_COO(inf, &nrows, &ncols, &nnz, &row, &col, &val);
    
    int nthreads = omp_get_max_threads();
    CSRPlusMatrix_t CSRP = NULL;
    CSRP_init_with_COO_matrix(nrows, ncols, nnz, row, col, val, &CSRP);
    
    double *rhs = (double*) malloc(sizeof(double) * nrows);
    double *sol = (double*) malloc(sizeof(double) * ncols);
    assert(rhs != NULL && sol != NULL);
    
    memset(sol, 0, sizeof(double) * ncols);
    for (int i = 0; i < nrows; i++)
        rhs[i] = (double) (i % 114);

    // Initialize MAGMA
    magma_init();
    magma_dopts   opts;
    magma_queue_t queue;
    magma_queue_create(0, &queue);
    
    magma_d_matrix A = {Magma_CSR}, dA = {Magma_CSR};
    magma_d_matrix b = {Magma_CSR}, db = {Magma_CSR};
    magma_d_matrix x = {Magma_CSR}, dx = {Magma_CSR};
    
    // Pass the system to MAGMA
    magma_dcsrset(nrows, ncols, CSRP->row_ptr, CSRP->col, CSRP->val, &A, queue);
    magma_dvset(nrows, 1, rhs, &b, queue);
    magma_dvset(ncols, 1, sol, &x, queue);
    
    // Setup MAGMA solver
    opts.solver_par.solver  = Magma_CG;
    opts.solver_par.maxiter = 10000;
    opts.solver_par.rtol    = 1e-6;
    
    // Initialize MAGMA solver
    magma_dsolverinfo_init(&opts.solver_par, &opts.precond_par, queue);
    
    // Copy the system to the device (optional, only necessary if using the GPU)
    magma_dmtransfer(A, &dA, Magma_CPU, Magma_DEV, queue);
    magma_dmtransfer(b, &db, Magma_CPU, Magma_DEV, queue);

    // Solve the linear system
    for (int i = 0; i < ntest; i++)
    {
        // Use zero initial guess
        magma_dmtransfer(x, &dx, Magma_CPU, Magma_DEV, queue);
        
        magma_d_solver(dA, db, &dx, &opts, queue);
        printf("MAGMA CG solve done, num_iter = %d, runtime = %lf (s), ", opts.solver_par.numiter, opts.solver_par.runtime);
        printf("init_res = %e, final_res = %e\n", opts.solver_par.init_res, opts.solver_par.final_res);
    }
    
    // Copy the solution back to the host
    magma_dmfree(&x, queue);
    magma_dmtransfer(dx, &x, Magma_CPU, Magma_DEV, queue);
    
    // Copy the solution in MAGMA host structure to the application code
    int one = 1;
    magma_dvget(x, &ncols, &one, &sol, queue);
    
    // Free the allocated memory and finalize MAGMA
    magma_dmfree(&dx, queue);
    magma_dmfree(&db, queue);
    magma_dmfree(&dA, queue);
    magma_queue_destroy(queue);
    magma_finalize();
    
    // Free host arrays
    free(row);
    free(col);
    free(val);
    free(rhs);
    free(sol);
    CSRP_free(CSRP);
    
    return 0;
}
