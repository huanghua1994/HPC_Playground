#ifndef __FD3D_H__
#define __FD3D_H__

// Coefficients for 2nd derivatives
const static double stencil_coefs[7] = {
 -7.09617303612850,
  4.07836887902444,
 -0.637245137347569,
  0.125875582685939,
 -0.0212415045782523,
  0.00247173871456027,
 -0.000143040434870386
};

// Set parameters for Laplacian operator
// Input parameters:
//    nx, ny, nz : Sizes of x, y, z direction
//    radius     : Radius of the stencil
//    x_in       : Input vector
void FD3D_Laplacian_set_param(const int nx_, const int ny_, const int nz_, const int radius_);

// Laplacian operator
// Input parameters:
//    x_in  : Input vector
// Output parameters:
//    x_out : Output vector, := Laplacian * x_in
void FD3D_Laplacian_MatVec(const double *x_in, double *x_out);

#endif
