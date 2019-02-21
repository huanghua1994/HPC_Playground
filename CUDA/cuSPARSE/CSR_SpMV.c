#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>
//#include <mkl.h>

#include "CSRPlus.h"
#include "cuSPARSE_SpMV_test.h"

CSRPlusMatrix_t CSRP = NULL;

static void CSR_SpMV_CPU_ref(CSRPlusMatrix_t CSRP, const double * __restrict x, double * __restrict y)
{
    /*
    int    *row_ptr = CSRP->row_ptr;
    int    *col     = CSRP->col;
    double *val     = CSRP->val;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int irow = 0; irow < CSRP->nrows; irow++)
    {
        double res = 0.0;
        #pragma omp simd
        for (int idx = row_ptr[irow]; idx < row_ptr[irow + 1]; idx++)
            res += val[idx] * x[col[idx]];
        y[irow] = res;
    }
    */
    
    CSRP_SpMV(CSRP, x, y);
}

static double getL2NormError(const double *__restrict x, const double *__restrict y, const int n)
{
    double res = 0.0;
    
    #pragma omp simd
    for (int i = 0; i < n; i++)
    {
        double diff = fabs(x[i] - y[i]);
        res += diff * diff;
        if (diff / fabs(x[i]) > 0.001) printf("%d: %lf %lf\n", i, x[i], y[i]);
    }
    res = sqrt(res);
    
    return res;
}

int main(int argc, char **argv)
{
    int *row, *row_ptr, *col;
    double *val, *x, *y0, *y1;
    
    int nrows, ncols, nnz, ntest;
    if (argc < 3) 
    {
        printf("Usage: CSR_SpMV <nrows> <ncols> <nnz>\n");
        assert(argc >= 3);
    }
    
    nrows = atoi(argv[1]);
    ncols = atoi(argv[2]);
    nnz   = atoi(argv[3]);
    
    ntest = (2000000000 / nnz) + 1;
    if (ntest > 4000) ntest = 4000;
    
    row     = (int*)    malloc(sizeof(int)    * nnz);
    col     = (int*)    malloc(sizeof(int)    * nnz);
    val     = (double*) malloc(sizeof(double) * nnz);
    x       = (double*) malloc(sizeof(double) * ncols);
    y0      = (double*) malloc(sizeof(double) * nrows);
    y1      = (double*) malloc(sizeof(double) * nrows);
    assert(row     != NULL);
    assert(col     != NULL);
    assert(val     != NULL);
    assert(x       != NULL);
    assert(y0      != NULL);
    assert(y1      != NULL);
    
    for (int i = 0; i < ncols; i++) x[i] = (double) (i % 1919);
    
    int nthreads = omp_get_max_threads();
    
    // Generate random matrix
    srand(time(NULL));
    //#pragma omp parallel for
    for (int i = 0; i < nnz; i++) 
    {
        row[i] = rand() % nrows;
        col[i] = rand() % ncols;
        val[i] = 1.0;
    }
    CSRP_init_with_COO_matrix(nrows, nnz, row, col, val, &CSRP);
    CSRP_partition(nthreads, CSRP);
    CSRP_optimize_NUMA(CSRP);
    row_ptr = CSRP->row_ptr;
    
    double st, et, ut, GFlops;
    GFlops = 2.0 * (double) nnz / 1000000000.0;

    // Get reference result
    st = omp_get_wtime();
    for (int k = 0; k < ntest; k++)
        CSR_SpMV_CPU_ref(CSRP, x, y0);
    et = omp_get_wtime();
    ut = (et - st) / (double) ntest;
    printf("Reference OMP CSR SpMV done, used time = %lf (ms), %lf GFlops\n", ut * 1000.0, GFlops / ut);

    // Get test result
    cuSPARSE_SpMV_test(nrows, ncols, nnz, CSRP->row_ptr, CSRP->col, CSRP->val, x, y1, ntest);
    /*
    sparse_matrix_t mat;
    struct matrix_descr mat_descr;
    mat_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mat_descr.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr(
		&mat, SPARSE_INDEX_BASE_ZERO, nrows, ncols, 
		row_idx, row_idx + 1, col, val
	);
    mkl_sparse_set_mv_hint(mat, SPARSE_OPERATION_NON_TRANSPOSE, mat_descr, ntest); 
    mkl_sparse_optimize(mat);
    mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mat, 
        mat_descr, x, 0.0, y1
    );
    */

    double err = getL2NormError(y0, y1, nrows);
    printf("|| y_ref - y ||_2 = %e\n", err);
    
    free(row);
    free(col);
    free(val);
    free(x);
    free(y0);
    free(y1);
}
