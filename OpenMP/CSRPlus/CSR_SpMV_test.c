#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include <mkl.h>
#include <mkl_spblas.h>

#include "mmio.h"
#include "utils.h"
#include "CSRPlus.h"

void read_mtx_file_to_COO_mat(
    const char *fname, int *nrow_, int *ncol_, int *nnz_,
    int **row_, int **col_, double **val_
)
{
    FILE *inf = fopen(fname, "r");
    if (inf == NULL)
    {
        printf("Failed to open file %s.\n", fname);
        exit(1);
    }
    
    MM_typecode matcode;
    if (mm_read_banner(inf, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    
    int nrow, ncol, nnz0, nnz;
    int is_symm = mm_is_symmetric(matcode);
    int is_comp = mm_is_complex(matcode);
    int is_patt = mm_is_pattern(matcode);
    
    int ret_code = mm_read_mtx_crd_size(inf, &nrow, &ncol, &nnz0);
    assert(ret_code == 0);
    nnz = (is_symm) ? nnz0 * 2 : nnz0;
    
    int    *row = (int*)    malloc(sizeof(int)    * nnz);
    int    *col = (int*)    malloc(sizeof(int)    * nnz);
    double *val = (double*) malloc(sizeof(double) * nnz);
    assert((row != NULL) && (col != NULL) && (val != NULL));
    
    printf("Loading matrix %s: %d rows, %d columns\n", fname, nrow, ncol);
    printf("Matrix: symmetric = %d, complex = %d, pattern = %d\n", is_symm, is_comp, is_patt);
    
    int irow, icol;
    double ival, imag;
    
    // General real / integer matrix
    if ((is_patt == 0) && (is_symm == 0) && (is_comp == 0))
    {
        for (int i = 0; i < nnz0; i++)
        {
            fscanf(inf, "%d",  &irow);
            fscanf(inf, "%d",  &icol);
            fscanf(inf, "%lf", &ival);
            row[i] = irow - 1;
            col[i] = icol - 1;
            val[i] = ival;
        }
    }
    
    // General complex matrix, use only the real part
    if ((is_patt == 0) && (is_symm == 0) && (is_comp == 1))
    {
        for (int i = 0; i < nnz0; i++)
        {
            fscanf(inf, "%d",  &irow);
            fscanf(inf, "%d",  &icol);
            fscanf(inf, "%lf", &ival);
            fscanf(inf, "%lf", &imag);
            row[i] = irow - 1;
            col[i] = icol - 1;
            val[i] = ival;
        }
    }
    
    // Symmetric real / integer matrix, read once add twice
    if ((is_patt == 0) && (is_symm == 1) && (is_comp == 0))
    {
        nnz = 0;
        for (int i = 0; i < nnz0; i++)
        {
            fscanf(inf, "%d",  &irow);
            fscanf(inf, "%d",  &icol);
            fscanf(inf, "%lf", &ival);
            row[nnz] = irow - 1;
            col[nnz] = icol - 1;
            val[nnz] = ival;
            nnz++;
            if (irow != icol)
            {
                row[nnz] = icol - 1;
                col[nnz] = irow - 1;
                val[nnz] = ival;
                nnz++;
            }
        }
    }
    
    // Symmetric complex matrix, read once add twice, use only the real part
    if ((is_patt == 0) && (is_symm == 1) && (is_comp == 1))
    {
        nnz = 0;
        for (int i = 0; i < nnz0; i++)
        {
            fscanf(inf, "%d",  &irow);
            fscanf(inf, "%d",  &icol);
            fscanf(inf, "%lf", &ival);
            fscanf(inf, "%lf", &imag);
            row[nnz] = irow - 1;
            col[nnz] = icol - 1;
            val[nnz] = ival;
            nnz++;
            if (irow != icol)
            {
                row[nnz] = icol - 1;
                col[nnz] = irow - 1;
                val[nnz] = ival;
                nnz++;
            }
        }
    }
    
    // General pattern matrix, use 1 as value
    if ((is_patt == 1) && (is_symm == 0))
    {
        for (int i = 0; i < nnz0; i++)
        {
            fscanf(inf, "%d",  &irow);
            fscanf(inf, "%d",  &icol);
            row[i] = irow - 1;
            col[i] = icol - 1;
            val[i] = 1.0;
        }
    }
    
    // Symmetric pattern matrix, use 1 as value, read once add twice
    if ((is_patt == 1) && (is_symm == 1))
    {
        nnz = 0;
        for (int i = 0; i < nnz0; i++)
        {
            fscanf(inf, "%d",  &irow);
            fscanf(inf, "%d",  &icol);
            row[nnz] = irow - 1;
            col[nnz] = icol - 1;
            val[nnz] = 1.0;
            nnz++;
            if (irow != icol)
            {
                row[nnz] = icol - 1;
                col[nnz] = irow - 1;
                val[nnz] = 1.0;
                nnz++;
            }
        }
    }
    
    printf("Load matrix done, %d non-zeros created\n\n", nnz);

    *nrow_ = nrow;
    *ncol_ = ncol;
    *nnz_  = nnz;
    *row_  = row;
    *col_  = col;
    *val_  = val;
}

void test_CSRPlus_SpMV(CSRP_mat_t csrp, const int ntest, const double *x, double *y)
{
    double st, et, ut;
    double GFlops = 2e-9 * (double) csrp->nnz;
    
    st = get_wtime_sec();
    int nthread = omp_get_max_threads();
    CSRP_partition_multithread(csrp, nthread, nthread);
    CSRP_optimize_NUMA(csrp);
    et = get_wtime_sec();
    ut = et - st;
    printf("CSRPlus setup and optimization done, used time = %.3lf (ms) \n", ut * 1000.0);
    
    // Warm up
    CSRP_SpMV(csrp, x, y);
    // Real performance test
    st = get_wtime_sec();
    for (int k = 0; k < ntest; k++)
        CSRP_SpMV(csrp, x, y);
    et = get_wtime_sec();
    ut = (et - st) / (double) ntest;
    printf("CSRPlus SpMV done, used time = %.3lf (ms), %lf GFlops\n\n", ut * 1000.0, GFlops / ut);
}

void test_MKL_IE_SpMV(CSRP_mat_t csrp, const int ntest, const double *x, double *y)
{
    double st, et, ut;
    double GFlops = 2e-9 * (double) csrp->nnz;
    
    sparse_matrix_t mkl_sp_mat;
    struct matrix_descr mat_descr;
    mat_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mat_descr.diag = SPARSE_DIAG_NON_UNIT;
    st = get_wtime_sec();
    mkl_sparse_d_create_csr(
        &mkl_sp_mat, SPARSE_INDEX_BASE_ZERO, csrp->nrow, csrp->ncol, 
        csrp->row_ptr, csrp->row_ptr + 1, csrp->col, csrp->val
    );
    mkl_sparse_set_mv_hint(mkl_sp_mat, SPARSE_OPERATION_NON_TRANSPOSE, mat_descr, ntest); 
    mkl_sparse_optimize(mkl_sp_mat);
    et = get_wtime_sec();
    ut = et - st;
    printf("MKL I-E SpMV setup and optimization done, used time = %.3lf (ms) \n", ut * 1000.0);
    
    // Warm up
    mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_sp_mat, 
        mat_descr, x, 0.0, y
    );
    // Real performance test
    st = get_wtime_sec();
    for (int k = 0; k < ntest; k++)
    {
        mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_sp_mat, 
            mat_descr, x, 0.0, y
        );
    }
    et = get_wtime_sec();
    ut = (et - st) / (double) ntest;
    printf("MKL I-E SpMV done, used time = %.3lf (ms), %lf GFlops\n\n", ut * 1000.0, GFlops / ut);
}

int main(int argc, char **argv)
{
    if (argc < 2) 
    {
        printf("Usage: CSR_SpMV.x <mtx_file>\n");
        exit(1);
    }
    
    int nrow, ncol, nnz;
    int *row, *col;
    double *val;
    read_mtx_file_to_COO_mat(argv[1], &nrow, &ncol, &nnz, &row, &col, &val);
    
    int ntest = (2000000000 / nnz) + 1;
    if (ntest > 4000) ntest = 4000;
    
    double *x  = (double*) malloc(sizeof(double) * ncol);
    double *y0 = (double*) malloc(sizeof(double) * nrow);
    double *y1 = (double*) malloc(sizeof(double) * nrow);
    assert(x  != NULL);
    assert(y0 != NULL);
    assert(y1 != NULL);
    
    srand48(19241112);
    for (int i = 0; i < ncol; i++) x[i] = drand48() - 0.5;
    
    CSRP_mat_t csrp;
    CSRP_init_with_COO_mat(nrow, ncol, nnz, row, col, val, &csrp);

    // Get CSRPlus result
    test_CSRPlus_SpMV(csrp, ntest, x, y1);

    // Get MKL Inspector-Executor SpMV test result
    test_MKL_IE_SpMV(csrp, ntest, x, y0);

    double y0_2norm, err_2norm;
    calc_err_2norm(ncol, y0, y1, &y0_2norm, &err_2norm);
    printf("Result relative error = %e\n", err_2norm / y0_2norm);
    
    free(row);
    free(col);
    free(val);
    free(x);
    free(y0);
    free(y1);
    
    return 0;
}
