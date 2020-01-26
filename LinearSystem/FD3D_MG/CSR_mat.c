#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <omp.h>

#include "CSR_mat.h"

void CSR_mat_init(const int nrow, const int ncol, const int max_nnz, CSR_mat_t *csr_mat_)
{
    CSR_mat_t csr_mat = (CSR_mat_t) malloc(sizeof(struct CSR_mat_));
    csr_mat->nrow = nrow;
    csr_mat->ncol = ncol;
    csr_mat->nnz  = 0;
    csr_mat->row_ptr = (int*)    malloc(sizeof(int)    * (nrow + 1));
    csr_mat->col     = (int*)    malloc(sizeof(int)    * max_nnz);
    csr_mat->val     = (double*) malloc(sizeof(double) * max_nnz);
    assert(csr_mat->row_ptr != NULL);
    assert(csr_mat->col     != NULL);
    assert(csr_mat->val     != NULL);
    *csr_mat_ = csr_mat;
}

void CSR_mat_destroy(CSR_mat_t csr_mat)
{
    if (csr_mat == NULL) return;
    free(csr_mat->row_ptr);
    free(csr_mat->col);
    free(csr_mat->val);
    free(csr_mat);
}

void CSR_SpMV(CSR_mat_t csr_mat, const double *x, double *y)
{
    int    nrow     = csr_mat->nrow;
    int    *row_ptr = csr_mat->row_ptr;
    int    *col     = csr_mat->col;
    double *val     = csr_mat->val;
    // Here we assume that no such row that has a lot of nnz
    #pragma omp parallel for schedule(dynamic, 16)
    for (int irow = 0; irow < nrow; irow++)
    {
        double res = 0.0;
        #pragma omp simd
        for (int i = row_ptr[irow]; i < row_ptr[irow + 1]; i++)
            res += val[i] * x[col[i]];
        y[irow] = res;
    }
}

void CSR_mat_transpose(CSR_mat_t csr_mat, CSR_mat_t *csr_matT_)
{
    CSR_mat_t csr_matT;
    CSR_mat_init(csr_mat->ncol, csr_mat->nrow, csr_mat->nnz, &csr_matT);
    csr_matT->nnz = csr_mat->nnz;

    int *nnz_row = (int*) malloc(sizeof(int) * csr_mat->ncol);
    assert(nnz_row != NULL);
    memset(nnz_row, 0, sizeof(int) * csr_mat->ncol);
    for (int i = 0; i < csr_mat->nnz; i++)
        nnz_row[csr_mat->col[i]]++;

    csr_matT->row_ptr[0] = 0;
    for (int i = 0; i < csr_matT->nrow; i++)
        csr_matT->row_ptr[i + 1] = csr_matT->row_ptr[i] + nnz_row[i];

    for (int irow = 0; irow < csr_mat->nrow; irow++)
    {
        for (int i = csr_mat->row_ptr[irow]; i < csr_mat->row_ptr[irow + 1]; i++)
        {
            int icol = csr_mat->col[i];
            int Tidx = csr_matT->row_ptr[icol];
            csr_matT->col[Tidx] = irow;
            csr_matT->val[Tidx] = csr_mat->val[i];
            csr_matT->row_ptr[icol]++;
        }
    }
    for (int irow = csr_matT->nrow; irow >= 1; irow--)
        csr_matT->row_ptr[irow] = csr_matT->row_ptr[irow - 1];
    csr_matT->row_ptr[0] = 0;

    *csr_matT_ = csr_matT;
    free(nnz_row);
}

void CSR_mat_to_dense_mat(CSR_mat_t csr_mat, double **mat_)
{
    int nrow = csr_mat->nrow;
    int ncol = csr_mat->ncol;
    int *row_ptr = csr_mat->row_ptr;
    int *col = csr_mat->col;
    double *val = csr_mat->val;
    double *mat = (double*) malloc(sizeof(double) * nrow * ncol);
    assert(mat != NULL);

    #pragma omp parallel for 
    for (int i = 0; i < nrow * ncol; i++) mat[i] = 0.0;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int irow = 0; irow < nrow; irow++)
    {
        double *mat_irow = mat + irow * ncol;
        #pragma omp simd
        for (int i = row_ptr[irow]; i < row_ptr[irow + 1]; i++)
            mat_irow[col[i]] = val[i];
    }

    *mat_ = mat;
}
