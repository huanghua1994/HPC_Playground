#ifndef __CSR_MAT_H__
#define __CSR_MAT_H__

struct CSR_mat_
{
    int    nrow, ncol, nnz;
    int    *row_ptr;
    int    *col;
    double *val;
};

typedef struct CSR_mat_* CSR_mat_t;

#ifdef __cplusplus
extern "C" {
#endif

void CSR_mat_init(const int nrow, const int ncol, const int max_nnz, CSR_mat_t *csr_mat_);

void CSR_mat_destroy(CSR_mat_t csr_mat);

void CSR_SpMV(CSR_mat_t csr_mat, const double *x, double *y);

void CSR_mat_transpose(CSR_mat_t csr_mat, CSR_mat_t *csr_matT_);

void CSR_mat_to_dense_mat(CSR_mat_t csr_mat, double **mat_);

#ifdef __cplusplus
};
#endif

#endif
