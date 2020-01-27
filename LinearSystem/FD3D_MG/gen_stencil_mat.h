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

// Generate the restriction matrix R, tri-linear prolongation matrix P, and
// smoother diagonal matrix M for a level using a FD Laplacian matrix
// Input parameters:
//   Nx, Ny, Nz    : Number of FD grid points on x, y, z directions
//   BCx, BCy, BCz : Boundary condition on x, y, z directions, 0 : periodic, 1 : Dirichlet
//   A             : The FD Laplacian matrix
// Output parameters:
//   *R_ : Restriction matrix, R = 0.125 * P^T
//   *P_ : Tri-linear prolongation matrix
//   *M_ : Smoother diagonal matrix, M = 0.75 ./ diag(R * A * P)
void gen_R_P_diag_RAP(
    const int Nx,  const int Ny,  const int Nz, 
    const int BCx, const int BCy, const int BCz,
    CSR_mat_t A,   CSR_mat_t *R_, CSR_mat_t *P_, double **M_
);

#ifdef __cplusplus
};
#endif

#endif 
