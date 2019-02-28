#ifndef __CUSPARSE_SPMV_TEST_H__
#define __CUSPARSE_SPMV_TEST_H__

#ifdef __cplusplus
extern "C" {
#endif

void cuSPARSE_SpMV_test(
    const int nrows, const int ncols, const int nnz, 
    const int *row_ptr, const int *col, const double *val, 
    const double *x, double *y, const int ntest
);

void cuSPARSE_SpMM_test(
    const int nrows, const int ncols, const int nnz, const int X_ncol, 
    const int *row_ptr, const int *col, const double *val, 
    const double *X, const int ldX, double *Y, const int ldY, const int ntest
);

#ifdef __cplusplus
}
#endif

#endif

