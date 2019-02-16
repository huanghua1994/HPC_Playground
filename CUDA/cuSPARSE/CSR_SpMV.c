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

static int cmp_pair(int M1, int N1, int M2, int N2)
{
    if (M1 == M2) return (N1 < N2);
    else return (M1 < M2);
}

static void qsortCOO2CSR(int *row, int *col, double *val, int l, int r)
{
    int i = l, j = r, row_tmp, col_tmp;
    int mid_row = row[(l + r) / 2];
    int mid_col = col[(l + r) / 2];
    double val_tmp;
    while (i <= j)
    {
        while (cmp_pair(row[i], col[i], mid_row, mid_col)) i++;
        while (cmp_pair(mid_row, mid_col, row[j], col[j])) j--;
        if (i <= j)
        {
            row_tmp = row[i]; row[i] = row[j]; row[j] = row_tmp;
            col_tmp = col[i]; col[i] = col[j]; col[j] = col_tmp;
            val_tmp = val[i]; val[i] = val[j]; val[j] = val_tmp;
            
            i++;  j--;
        }
    }
    if (i < r) qsortCOO2CSR(row, col, val, i, r);
    if (j > l) qsortCOO2CSR(row, col, val, l, j);
}

static void compressIndices(int *idx, int *idx_ptr, int nindex, int nelem)
{
    int curr_pos = 0, end_pos;
    idx_ptr[0] = 0;
    for (int index = 0; index < nindex; index++)
    {
        for (end_pos = curr_pos; end_pos < nelem; end_pos++)
            if (idx[end_pos] > index) break;
        idx_ptr[index + 1] = end_pos;
        curr_pos = end_pos;
    }
    idx_ptr[nindex] = nelem; 
}

static void COO2CSR(int *row, int *row_idx, int *col, double *val, int nnz, int nrows)
{
    qsortCOO2CSR(row, col, val, 0, nnz - 1);
    compressIndices(row, row_idx, nrows, nnz);
}

CSRP_blk_info_t CSRP_blkinfo = NULL;

static void CSR_SpMV_CPU_ref(
    const int *row_idx, const int *col, const double *val, 
    const int nrows, const double * __restrict x, double * __restrict y
)
{
    /*
    #pragma omp parallel for schedule(dynamic, 16)
    for (int irow = 0; irow < nrows; irow++)
    {
        double res = 0.0;
        #pragma omp simd
        for (int idx = row_idx[irow]; idx < row_idx[irow + 1]; idx++)
            res += val[idx] * x[col[idx]];
        y[irow] = res;
    }
    */
    
    int nnz = row_idx[nrows];
    int nthreads = omp_get_max_threads();
    int nblocks  = nthreads;
    
    if (CSRP_blkinfo == NULL)  // Initialize it once, since we are using the same matrix
    {
        CSRP_blkinfo = (CSRP_blk_info_t) malloc(sizeof(CSRP_blk_info) * nthreads);
        CSRP_equal_nnz_partition(nnz, nblocks, nrows, row_idx, CSRP_blkinfo);
    }
    
    CSRP_SpMV(row_idx, col, val, nrows, nblocks, CSRP_blkinfo, x, y);
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
    int *row, *row_idx, *col;
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
    row_idx = (int*)    malloc(sizeof(int)    * (nrows + 1));
    val     = (double*) malloc(sizeof(double) * nnz);
    x       = (double*) malloc(sizeof(double) * ncols);
    y0      = (double*) malloc(sizeof(double) * nrows);
    y1      = (double*) malloc(sizeof(double) * nrows);
    assert(row     != NULL);
    assert(col     != NULL);
    assert(row_idx != NULL);
    assert(val     != NULL);
    assert(x       != NULL);
    assert(y0      != NULL);
    assert(y1      != NULL);
    
    for (int i = 0; i < ncols; i++) x[i] = (double) (i % 1919);
    
    // Generate random matrix
    srand(time(NULL));
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) 
    {
        row[i] = rand() % nrows;
        col[i] = rand() % ncols;
        val[i] = 1.0;
    }
    COO2CSR(row, row_idx, col, val, nnz, nrows);
    printf("Generating random matrix done\n");
    
    double st, et, ut, GFlops;
    GFlops = 2.0 * (double) nnz / 1000000000.0;
    
    // Get reference result
    st = omp_get_wtime();
    for (int k = 0; k < ntest; k++)
    {
        CSR_SpMV_CPU_ref(row_idx, col, val, nrows, x, y0);
        val[k] = 1.0;
    }
    et = omp_get_wtime();
    ut = (et - st) / (double) ntest;
    printf("Reference OMP CSR SpMV done, used time = %lf (ms), %lf GFlops\n", ut * 1000.0, GFlops / ut);

    // Get test result
    cuSPARSE_SpMV_test(nrows, ncols, nnz, row_idx, col, val, x, y1, ntest);
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
    free(row_idx);
    free(val);
    free(x);
    free(y0);
    free(y1);
}