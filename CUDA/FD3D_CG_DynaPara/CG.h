#ifndef __CG_H__
#define __CG_H__

// Classic Conjugate Gradients method for solving Laplacian * x = b
// Input parameters:
//    n        : Size of the Laplacian matrix
//    res_tol  : Relative residual tolerance: stop when ||r||_2 * res_tol <= ||b||_2
//    max_iter : Maximum step of iteration
//    b        : Right-hand side vector
// Output parameters:
//    x        : Solution vector
void CG_classic(
    const int n, const double res_tol, const int max_iter,
    const double *b, double *x
);

#endif
