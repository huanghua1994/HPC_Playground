#ifndef __CG_H__
#define __CG_H__

#include <mkl.h>
#include <mkl_spblas.h>

// Classic Conjugate Gradients method for solving A * x = b
// Input parameters:
//   A        : Target matrix
//   n        : Size of matrix A
//   res_tol  : Relative residual tolerance: stop when ||r||_2 * res_tol <= ||b||_2
//   max_iter : Maximum step of iteration
//   b        : Right-hand side vector
// Output parameters:
//   x : Solution vector
void CG_classic(
    sparse_matrix_t A, const int n, const double res_tol, 
    const int max_iter, const double *b, double *x
);

#endif
