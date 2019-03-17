#ifndef __FD3D_H__
#define __FD3D_H__

// Coefficients for 2nd derivatives
const static double stencil_coefs[7] = {
-73.6998373668631217,
 42.3573553767459572,
 -6.6183367776165563,
  1.3073257832328999,
 -0.2206112259205519,
  0.0256711244707551,
 -0.0014855974809465
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
