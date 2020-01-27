#ifndef __GEN_STENCIL_MAT_H__
#define __GEN_STENCIL_MAT_H__

#include "CSR_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

// Generate Laplacian matrix using finite difference on an orthogonal grid
// Input parameters:
//   Lx, Ly, Lz    : Length on x, y, z directions of the FD grid
//   Nx, Ny, Nz    : Number of FD grid points on x, y, z directions
//   BCx, BCy, BCz : Boundary condition on x, y, z directions, 0 : periodic, 1 : Dirichlet
//   FDn           : Finite difference radius
// Output parameter:
//   *A_ : The finite difference Laplacian matrix
void gen_fd_Lap_orth(
    const double Lx,  const int Nx, const int BCx,
    const double Ly,  const int Ny, const int BCy,
    const double Lz,  const int Nz, const int BCz,
    const int FDn, CSR_mat_t *A_
);

// Generate the restriction matrix R and tri-linear prolongation matrix P
// Input parameters:
//   Nx, Ny, Nz    : Number of FD grid points on x, y, z directions
//   BCx, BCy, BCz : Boundary condition on x, y, z directions, 0 : periodic, 1 : Dirichlet
// Output parameters:
//   *R_ : Tri-linear restriction matrix
//   *P_ : Tri-linear prolongation matrix, P = 8 * R^T
void gen_trilin_R_P(
    const int Nx,  const int Ny,  const int Nz, 
    const int BCx, const int BCy, const int BCz,
    CSR_mat_t *R_, CSR_mat_t *P_
);

#ifdef __cplusplus
};
#endif

#endif 
