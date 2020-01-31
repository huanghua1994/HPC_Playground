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
//   cell_dims  : Size 3, length of the FD domain on x, y, z directions
//   grid_sizes : Size 3, number of the finite difference grid points on x, y, z directions
//   BCs        : Size 3, boundary condition on x, y, z directions, 0 : periodic, 1 : Dirichlet.
//                For each direction, the mesh space == cell_dims(k) / (grid_sizes(k) - BCs(k)).
//   FDn        : Finite difference radius
void FD3D_Laplacian_set_param(const double *cell_dims, const int *grid_sizes, const int *BCs, const int FDn);

// Laplacian operator
// Input parameters:
//   x_in  : Input vector
// Output parameters:
//   x_out : Output vector, := Laplacian * x_in
void FD3D_Laplacian_MatVec(const double *x_in, double *x_out);

#endif
