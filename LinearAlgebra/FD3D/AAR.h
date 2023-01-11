#ifndef __AAR_H__
#define __AAR_H__

// Alternating Anderson-Richardson (AAR) method for solving Laplacian * x = b
// Input parameters:
//    N        : Size of the Laplacian matrix
//    res_tol  : Relative residual tolerance: stop when ||r||_2 * res_tol <= ||b||_2
//    max_iter : Maximum step of iteration
//    b        : Right-hand side vector
// Output parameters:
//    x        : Solution vector
void AAR(
    const int N, const double res_tol, const int max_iter,
    const double *b, double *x
);

#endif
