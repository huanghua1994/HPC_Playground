#ifndef __POISSON_MULTIGRID_H__
#define __POISSON_MULTIGRID_H__

#include "CSR_mat.h"

#include <mkl.h>
#include <mkl_spblas.h>

struct mg_data_
{
    int    nlevel;      // Finest grid is level 0, coarsest grid is level nlevel
    int    *vlen;       // Size nlevel+1, lengths of vectors on each level
    int    *lastA_ipiv; // Size vec_len[level], ipiv array from LAPACKE_dgetrf
    double *lastA_inv;  // Size vec_len[level]*vec_len[level], LU factorization from 
                        // LAPACKE_dgetrf or pseudo-inverse of the coarsest grid matrix
    double **ev;        // Size nlevel+1, ev[i] size vec_len[i], error vectors at each level
    double **rv;        // Size nlevel+1, rv[i] size vec_len[i], residual vectors at each level
    double **tv;        // Size nlevel+1, tv[i] size 2*vec_len[i], temporary vectors at each level
    double **M;         // Size nlevel+1, M[i] size vec_len[i]*vec_len[i], the smoother matrix 
                        // at each level. Here we use the scaled inverse diagonal of coefficient
                        // matrix at each level: M[i] = 0.75 ./ diag(R[i] * A[i] * P[i]) for i > 0, 
                        // and M[0] = 0.75 ./ diag(A[0]). 
    CSR_mat_t *A;       // Size nlevel+1, finite difference Laplacian matrix at each level, A[0] = A[1]
    CSR_mat_t *R;       // Size nlevel+1, restriction matrix at each level, R[i] = 0.125 * P[i]^T
    CSR_mat_t *P;       // Size nlevel+1, tri-linear prolongation matrix at each level
    
    sparse_matrix_t *mkl_sp_A;
    sparse_matrix_t *mkl_sp_R;
    sparse_matrix_t *mkl_sp_P;
};

typedef struct mg_data_* mg_data_t;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a mg_data data structure and construct matrices at each level
// Input parameters:
//   cell_dims  : Size 3, length of the FD domain on x, y, z directions
//   grid_sizes : Size 3, number of the finite difference grid points on x, y, z directions
//   BCs        : Size 3, boundary condition on x, y, z directions, 0 : periodic, 1 : Dirichlet.
//                For each direction, the mesh space == cell_dims(k) / (grid_sizes(k) - BCs(k)).
//   FDn        : Finite difference radius
// Output parameter:
//   *mg_data_ : Initialized mg_data structure with all matrices
void MG_init(
    const double *cell_dims, const int *grid_sizes, const int *BCs, 
    const int FDn, mg_data_t *mg_data_
);

// Destroy a mg_data data structure
// Input parameter:
//   mg_data : A mg_data structure to be destroyed 
void MG_destroy(mg_data_t mg_data);

// Solve the Poisson equation using multigrid 
// Input parameters:
//   mg_data : Initialized mg_data data structure
//   b       : Right-hand-side vector
//   reltol  : Relative residual error tolerance
// Output parameter:
//   x : Solution vector
void MG_solve(mg_data_t mg_data, const double *b, double *x, const double reltol);

#ifdef __cplusplus
};
#endif

#endif

