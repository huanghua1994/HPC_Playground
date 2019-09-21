#ifndef __KERNEL_PTR_H__
#define __KERNEL_PTR_H__

// Pointer to function that performs kernel matrix matvec using given sets of 
// points and given input vector. The kernel function must be symmetric.
// Input parameters:
//   coord0 : Matrix, size 3-by-ld0, coordinates of the 1st point set
//   ld0    : Leading dimension of coord0, should be >= n0
//   n0     : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1 : Matrix, size 3-by-ld1, coordinates of the 2nd point set
//   ld1    : Leading dimension of coord1, should be >= n1
//   n1     : Number of points in coord1 (each column in coord0 is a coordinate)
//   x_in   : Vector, size >= n1, will be left multiplied by kernel_matrix(coord0, coord1)
// Output parameter:
//   x_out : Vector, size >= n0, x_out += kernel_matrix(coord0, coord1) * x_in
typedef void (*kernel_matvec_fptr) (
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
);

#endif
