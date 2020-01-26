#ifndef __GEN_STENCIL_MAT_H__
#define __GEN_STENCIL_MAT_H__

#include "CSR_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

void gen_fd_Lap_orth(
    const double Lx,  const int Nx, const int BCx,
    const double Ly,  const int Ny, const int BCy,
    const double Lz,  const int Nz, const int BCz,
    const int FDn, CSR_mat_t *A_
);

void gen_R_P_diag_RAP(
    const int Nx,  const int Ny,  const int Nz, 
    const int BCx, const int BCy, const int BCz,
    CSR_mat_t A,   CSR_mat_t *R_, CSR_mat_t *P_, double **M_
);

#ifdef __cplusplus
};
#endif

#endif 
