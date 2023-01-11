#ifndef __POISSON_FFT_SOLVER__
#define __POISSON_FFT_SOLVER__

// Solve a Poisson equation -\nabla^2 u = f with period boundary condition.
// \nabla^2 is discretized with order-(2*radius) finite difference, and 
// the FD domain is a cube with equal mesh size on each direction
// Input parameters:
//   n{x, y, z}   : Number of grid points on x, y, z direction
//   radius       : FD radius
//   w2_{x, y, z} : FD stencil coefficients, length radius+1
//   f_rhs        : Right-hand side of the equation, length nx*ny*nz
// Output parameters:
//   u_sol        : Solution, length nx*ny*nz
void Poisson_FD3D_PBC_FFT_Solver(
    const int nx, const int ny, const int nz, const int radius, 
    const double *w2_x, const double *x2_y, const double *w2_z, 
    double *f_rhs, double *u_sol
);

#endif
