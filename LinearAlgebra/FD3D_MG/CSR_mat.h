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

// Initialize a CSR_mat data structure and allocate memory
// Input parameters:
//   nrow    : Number of rows
//   ncol    : Number of columns
//   max_nnz : Maximum number of non-zero values 
// Output parameter:
//   *csr_mat_ : Initialized CSR_mat data structure, note: no actual data inside
void CSR_mat_init(const int nrow, const int ncol, const int max_nnz, CSR_mat_t *csr_mat_);

// Destroy a CSR_mat data structure
// Input parameter:
//   csr_mat : A CSR_mat data structure to be destroyed
void CSR_mat_destroy(CSR_mat_t csr_mat);

// OpenMP parallelized CSR sparse matrix multiply dense vector
// Input parameters:
//   csr_mat : Initialized CSR_mat data structure with actual data
//   x       : Size csr_mat->ncol, input vector
// Output parameter:
//   y : Size csr_mat->nrow, output vector
void CSR_SpMV(CSR_mat_t csr_mat, const double *x, double *y);

// Transpose a CSR_mat data structure and get a new CSR_mat data structure
// Input parameter:
//   csr_mat : A CSR_mat data structure to be transposed
// Output parameter:
//   *csr_matT_ : The transpose of csr_mat, stored in a CSR_mat data structure
void CSR_mat_transpose(CSR_mat_t csr_mat, CSR_mat_t *csr_matT_);

// Export a CSR_mat data structure to a row-major dense matrix
// Input parameter:
//   csr_mat : A CSR_mat data structure to be exported
// Output parameter:
//   *mat_ : Size csr_mat->nrow * csr_mat->ncol, dense form of csr_mat
void CSR_mat_to_dense_mat(CSR_mat_t csr_mat, double **mat_);

#ifdef __cplusplus
};
#endif

#endif
