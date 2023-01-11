#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "gen_stencil_mat.h"

static double factorial(const int n)
{
    double res = 1.0;
    for (int i = 2; i <= n; i++)
        res *= (double) i;
    return res;
}

static int periodic_pos(const int ix1, const int Nx, const int BCx)
{
    int ix = -1;
    if ((ix1 >= 0)  && (ix1 < Nx)) ix = ix1;
    if ((ix1  < 0)  && (BCx == 0)) ix = ix1 + Nx;
    if ((ix1 >= Nx) && (BCx == 0)) ix = ix1 - Nx;
    return ix;
}

static void calc_fd_shift_pos(
    const int ix, const int Nx, const int BCx, 
    const int FDn, int *shift_ix
)
{
    for (int r = -FDn; r <= FDn; r++)
    {
        int shift_r = r + FDn;
        int ixr = ix + r;
        shift_ix[shift_r] = periodic_pos(ixr, Nx, BCx);
    }
}

void gen_fd_Lap_orth(
    const double Lx,  const int Nx, const int BCx,
    const double Ly,  const int Ny, const int BCy,
    const double Lz,  const int Nz, const int BCz,
    const int FDn, CSR_mat_t *A_
)
{
    // 1. Calculate the finite difference weights
    double *w2 = (double *) malloc(sizeof(double) * (FDn + 1));
    assert(w2 != NULL);
    double c1 = factorial(FDn);  c1 *= c1;
    for (int k = 1; k <= FDn; k++)
    {
        double c2 = ((k + 1) % 2) ? -2.0 : 2.0;
        double c3 = 1.0 / (double) (k * k);
        double c4 = factorial(FDn - k);
        double c5 = factorial(FDn + k);
        w2[k]  = c1 * c2 * c3 / (c4 * c5);
        w2[0] -= 2.0 * c3;
    }
    
    // 2. Combine the FD weights with mesh space
    int Nd = Nx * Ny * Nz;
    double inv_dx2 = Lx / (double) (Nx - BCx);
    double inv_dy2 = Ly / (double) (Ny - BCy);
    double inv_dz2 = Lz / (double) (Nz - BCz);
    inv_dx2 = 1.0 / (inv_dx2 * inv_dx2);
    inv_dy2 = 1.0 / (inv_dy2 * inv_dy2);
    inv_dz2 = 1.0 / (inv_dz2 * inv_dz2);
    double *coef_dxx = (double *) malloc(sizeof(double) * (2 * FDn + 1));
    double *coef_dyy = (double *) malloc(sizeof(double) * (2 * FDn + 1));
    double *coef_dzz = (double *) malloc(sizeof(double) * (2 * FDn + 1));
    assert(coef_dxx != NULL && coef_dyy != NULL && coef_dzz != NULL);
    for (int r = -FDn; r <= FDn; r++)
    {
        int shift_r = r + FDn;
        int abs_r = (r < 0) ? -r : r;
        coef_dxx[shift_r] = w2[abs_r] * inv_dx2;
        coef_dyy[shift_r] = w2[abs_r] * inv_dy2;
        coef_dzz[shift_r] = w2[abs_r] * inv_dz2;
    }
    double coef_0 = coef_dxx[FDn] + coef_dyy[FDn] + coef_dzz[FDn];
    
    // 3. Loop over each grid point
    int max_nnz = Nd * (6 * FDn + 1), nnz = 0;
    CSR_mat_t A;
    CSR_mat_init(Nd, Nd, max_nnz, &A);
    int    *row_ptr = A->row_ptr;
    int    *col     = A->col;
    double *val     = A->val;
    int *shift_iy = (int *) malloc(sizeof(int) * (2 * FDn + 1));
    int *shift_iz = (int *) malloc(sizeof(int) * (2 * FDn + 1));
    assert(shift_iy != NULL && shift_iz != NULL);
    for (int iz = 0; iz < Nz; iz++)
    {
        calc_fd_shift_pos(iz, Nz, BCz, FDn, shift_iz);
        for (int iy = 0; iy < Ny; iy++)
        {
            calc_fd_shift_pos(iy, Ny, BCy, FDn, shift_iy);
            for (int ix = 0; ix < Nx; ix++)
            {
                // (ix, iy, iz)
                int curr_row = ix + iy * Nx + iz * Nx * Ny;
                row_ptr[curr_row] = nnz;
                col[nnz] = curr_row;
                val[nnz] = coef_0;
                nnz++;
                // (ix +- r, iy, iz)
                // (ix, iy +- r, iz)
                // (ix, iy, iz +- r)
                for (int r = -FDn; r <= FDn; r++)
                {
                    if (r == 0) continue;
                    int shift_r = r + FDn;
                    int iyr = shift_iy[shift_r];
                    int izr = shift_iz[shift_r];
                    int ixr = periodic_pos(ix + r, Nx, BCx);
                    if (ixr != -1)
                    {
                        col[nnz] = ixr + iy * Nx + iz * Nx * Ny;
                        val[nnz] = coef_dxx[shift_r];
                        nnz++;
                    }
                    if (iyr != -1)
                    {
                        col[nnz] = ix + iyr * Nx + iz * Nx * Ny;
                        val[nnz] = coef_dyy[shift_r];
                        nnz++;
                    }
                    if (izr != -1)
                    {
                        col[nnz] = ix + iy * Nx + izr * Nx * Ny;
                        val[nnz] = coef_dzz[shift_r];
                        nnz++;
                    }
                }  // End of r loop
            }  // End of ix loop
        }  // End of iy loop
    }  // End of iz loop
    row_ptr[Nd] = nnz;
    A->nnz = nnz;
    
    *A_ = A;
    free(w2);
    free(coef_dxx);
    free(coef_dyy);
    free(coef_dzz);
    free(shift_iy);
    free(shift_iz);
}

const double pow_2_neg[4] = {1.0, 0.5, 0.25, 0.125};

static int gen_R_row_nnz(
    const int ix0, const int iy0, const int iz0,
    const int Nx,  const int Ny,  const int Nz,
    const int BCx, const int BCy, const int BCz,
    int *col,  double *val
)
{
    int nnz = 0;
    for (int iz1 = iz0 - 1; iz1 <= iz0 + 1; iz1++)
    {
        int iz = periodic_pos(iz1, Nz, BCz);
        if (iz == -1) continue;
        for (int iy1 = iy0 - 1; iy1 <= iy0 + 1; iy1++)
        {
            int iy = periodic_pos(iy1, Ny, BCy);
            if (iy == -1) continue;
            for (int ix1 = ix0 - 1; ix1 <= ix0 + 1; ix1++)
            {
                int ix = periodic_pos(ix1, Nx, BCx);
                if (ix == -1) continue;
                int dist = abs(iz1 - iz0) + abs(iy1 - iy0) + abs(ix1 - ix0);
                col[nnz] = ix + iy * Nx + iz * Nx * Ny;
                val[nnz] = 0.125 * pow_2_neg[dist];
                nnz++;
            }
        }
    }
    return nnz;
}

void gen_trilin_R_P(
    const int Nx,  const int Ny,  const int Nz, 
    const int BCx, const int BCy, const int BCz,
    CSR_mat_t *R_, CSR_mat_t *P_
)
{
    int Nd   = Nx * Ny * Nz;
    int M_Nx = Nx / 2;
    int M_Ny = Ny / 2;
    int M_Nz = Nz / 2;
    int M_Nd = M_Nx * M_Ny * M_Nz;
    
    CSR_mat_t R;
    CSR_mat_init(M_Nd, Nd, 27 * M_Nd, &R);
    int R_nnz = 0;
    int *R_row_ptr = R->row_ptr;
    int *R_col     = R->col;
    double *R_val  = R->val;
    for (int iz = 1; iz < Nz; iz += 2)
    {
        int M_iz = iz / 2;
        for (int iy = 1; iy < Ny; iy += 2)
        {
            int M_iy = iy / 2;
            for (int ix = 1; ix < Nx; ix += 2)
            {
                int M_ix  = ix / 2;
                int M_idx = M_ix + M_iy * M_Nx + M_iz * M_Nx * M_Ny;
                int nnz_ixyz = gen_R_row_nnz(
                    ix, iy, iz, Nx, Ny, Nz, BCx, BCy, BCz,
                    R_col + R_nnz, R_val + R_nnz
                );
                R_row_ptr[M_idx] = R_nnz;
                R_nnz += nnz_ixyz;
            }  // End of ix loop
        }  // End of iy loop
    }  // End of iz loop
    R_row_ptr[M_Nd] = R_nnz;
    R->nnz = R_nnz;
    
    CSR_mat_t P;
    CSR_mat_transpose(R, &P);
    #pragma omp simd
    for (int i = 0; i < P->nnz; i++)
        P->val[i] *= 8.0;
    
    *R_ = R;
    *P_ = P;
}

