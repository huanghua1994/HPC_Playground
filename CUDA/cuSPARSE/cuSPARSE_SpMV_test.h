#ifndef __CUSPARSE_SPMV_TEST_H__
#define __CUSPARSE_SPMV_TEST_H__

void cuSPARSE_SpMV_test(
    const int nrows, const int ncols, const int nnz, 
    const int *row_ptr, const int *col, const double *val, 
    const double *x, double *y, const int ntest
);

#endif
