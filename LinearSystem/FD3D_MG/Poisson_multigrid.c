#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>
#include <mkl_spblas.h>

#include "Poisson_multigrid.h"
#include "gen_stencil_mat.h"

void MG_destroy(mg_data_t mg_data)
{
    if (mg_data == NULL) return;
    for (int lvl = 0; lvl <= mg_data->nlevel; lvl++)  // Notice: <=
    {
        free(mg_data->ev[lvl]);
        free(mg_data->rv[lvl]);
        free(mg_data->tv[lvl]);
        free(mg_data->M[lvl]);
        CSR_mat_destroy(mg_data->A[lvl]);
        CSR_mat_destroy(mg_data->P[lvl]);
        CSR_mat_destroy(mg_data->R[lvl]);
    }
    free(mg_data->vlen);
    free(mg_data->ev);
    free(mg_data->rv);
    free(mg_data->tv);
    free(mg_data->lastA_inv);
    free(mg_data->lastA_ipiv);
    free(mg_data->M);
    free(mg_data->A);
    free(mg_data->P);
    free(mg_data->R);
}

void MG_init(
    const double *cell_dims, const int *grid_sizes, const int *BCs, 
    const int FDn, mg_data_t *mg_data_
)
{
    mg_data_t mg_data = (mg_data_t) malloc(sizeof(struct mg_data_));
    
    double Lx  = cell_dims[0];
    double Ly  = cell_dims[1];
    double Lz  = cell_dims[2];
    int    Nx  = grid_sizes[0];
    int    Ny  = grid_sizes[1];
    int    Nz  = grid_sizes[2];
    int    BCx = BCs[0];
    int    BCy = BCs[1];
    int    BCz = BCs[2];

    // 1. Calculate the number of levels we need
    int nlevel = 0, Nd;
    while ((Nx > 7) && (Ny > 7) && (Nz > 7))
    {
        Nx /= 2;  Ny /= 2;  Nz /= 2;
        nlevel++;
    }
    mg_data->nlevel = nlevel;
    int nlevel1 = nlevel + 1;
    mg_data->vlen = (int*)       malloc(sizeof(int)       * nlevel1);
    mg_data->ev   = (double**)   malloc(sizeof(double*)   * nlevel1);
    mg_data->rv   = (double**)   malloc(sizeof(double*)   * nlevel1);
    mg_data->tv   = (double**)   malloc(sizeof(double*)   * nlevel1);
    mg_data->M    = (double**)   malloc(sizeof(double*)   * nlevel1);
    mg_data->A    = (CSR_mat_t*) malloc(sizeof(CSR_mat_t) * nlevel1);
    mg_data->R    = (CSR_mat_t*) malloc(sizeof(CSR_mat_t) * nlevel1);
    mg_data->P    = (CSR_mat_t*) malloc(sizeof(CSR_mat_t) * nlevel1);

    // 2. Construct the finest grid 
    Nx = grid_sizes[0];
    Ny = grid_sizes[1];
    Nz = grid_sizes[2];
    Nd = Nx * Ny * Nz;
    mg_data->vlen[0] = Nd;
    // mg.A{1} = gen_fd_lap_orth(cell_dims, [Nx Ny Nz], BCs, FDn);
    gen_fd_Lap_orth(Lx, Nx, BCx, Ly, Ny, BCy, Lz, Nz, BCz, FDn, &mg_data->A[0]);
    CSR_mat_t A0 = mg_data->A[0];
    // mg.M{1} = ones(size(mg.A{1}, 1), 1) .* (0.75 / mg.A{1}(1, 1));
    double *M0 = (double*) malloc(sizeof(double) * Nd);
    assert(M0 != NULL);
    double scaled_diag = 0.75 / (A0->val[0]);
    for (int i = 0; i < Nd; i++) M0[i] = scaled_diag;
    mg_data->M[0] = M0;
    
    sparse_matrix_t mkl_sp_R, mkl_sp_A, mkl_sp_P;
    sparse_matrix_t mkl_sp_AP, mkl_sp_RAP;
    sparse_status_t ret;
    sparse_index_base_t indexing;
    int nrow, ncol, nnz, *rs, *re, *col;
    double *val;

    // 3. Construct the R, A, P matrices level by level
    int level = 0;
    while ((Nx > 7) && (Ny > 7) && (Nz > 7))
    {
        // mg.R{level} = gen_trilin_R([Nx Ny Nz], BCs);
        // mg.P{level} = 8 * mg.R{level}';
        gen_trilin_R_P(
            Nx, Ny, Nz, BCx, BCy, BCz,
            &mg_data->R[level], &mg_data->P[level]
        );
        
        // mg.M{level} = 0.75 ./ full(diag(mg.A{level}));
        double *M_lvl = (double*) malloc(sizeof(double) * mg_data->vlen[level]);
        assert(M_lvl != NULL);
        CSR_mat_t A_lvl = mg_data->A[level];
        for (int irow = 0; irow < mg_data->vlen[level]; irow++)
        {
            for (int j = A_lvl->row_ptr[irow]; j < A_lvl->row_ptr[irow + 1]; j++)
            {
                if (A_lvl->col[j] == irow)
                {
                    M_lvl[irow] = 0.75 / A_lvl->val[j];
                    break;
                }
            }
        }
        mg_data->M[level] = M_lvl;
        
        // mg.A{level+1} = mg.R{level} * mg.A{level} * mg.P{level};
        CSR_mat_t R_lvl  = mg_data->R[level];
        CSR_mat_t P_lvl  = mg_data->P[level];
        mkl_sparse_d_create_csr(
            &mkl_sp_R, SPARSE_INDEX_BASE_ZERO, R_lvl->nrow, R_lvl->ncol,
            R_lvl->row_ptr, R_lvl->row_ptr + 1, R_lvl->col, R_lvl->val
        );
        mkl_sparse_d_create_csr(
            &mkl_sp_A, SPARSE_INDEX_BASE_ZERO, A_lvl->nrow, A_lvl->ncol,
            A_lvl->row_ptr, A_lvl->row_ptr + 1, A_lvl->col, A_lvl->val
        );
        mkl_sparse_d_create_csr(
            &mkl_sp_P, SPARSE_INDEX_BASE_ZERO, P_lvl->nrow, P_lvl->ncol,
            P_lvl->row_ptr, P_lvl->row_ptr + 1, P_lvl->col, P_lvl->val
        );
        mkl_sparse_optimize(mkl_sp_R);
        mkl_sparse_optimize(mkl_sp_A);
        mkl_sparse_optimize(mkl_sp_P);
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, mkl_sp_A, mkl_sp_P,  &mkl_sp_AP);
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, mkl_sp_R, mkl_sp_AP, &mkl_sp_RAP);
        mkl_sparse_d_export_csr(mkl_sp_RAP, &indexing, &nrow, &ncol, &rs, &re, &col, &val);
        nnz = re[nrow - 1];
        CSR_mat_t A_lvl1;
        CSR_mat_init(nrow, ncol, nnz, &A_lvl1);
        memcpy(A_lvl1->row_ptr, rs,  sizeof(int)    * nrow);
        memcpy(A_lvl1->col,     col, sizeof(int)    * nnz);
        memcpy(A_lvl1->val,     val, sizeof(double) * nnz);
        A_lvl1->nnz = nnz;
        A_lvl1->row_ptr[nrow] = nnz;
        mg_data->A[level + 1] = A_lvl1;
        mkl_sparse_destroy(mkl_sp_R);
        mkl_sparse_destroy(mkl_sp_A);
        mkl_sparse_destroy(mkl_sp_P);
        mkl_sparse_destroy(mkl_sp_AP);
        mkl_sparse_destroy(mkl_sp_RAP);
        
        Nx /= 2;  Ny /= 2;  Nz /= 2; 
        Nd = Nx * Ny * Nz;
        level++;
        mg_data->vlen[level] = Nd;
    }  // End of "while ((Nx > 7) && (Ny > 7) && (Nz > 7))"
    mg_data->R[mg_data->nlevel] = NULL;
    mg_data->P[mg_data->nlevel] = NULL;
    mg_data->M[mg_data->nlevel] = NULL;
    
    // 4. Allocate error, residual, and tmp vectors for each level
    for (int level = 0; level <= nlevel; level++)
    {
        size_t lvl_vlen = sizeof(double) * mg_data->vlen[level];
        mg_data->ev[level] = (double*) malloc(lvl_vlen);
        mg_data->rv[level] = (double*) malloc(lvl_vlen);
        mg_data->tv[level] = (double*) malloc(lvl_vlen * 2);
        assert(mg_data->ev[level] != NULL);
        assert(mg_data->rv[level] != NULL);
        assert(mg_data->tv[level] != NULL);
    }
    
    // 5. Calculate the LU decomposition or pseudo-inverse for the 
    //    coarsest grid matrix R * A * P
    int last_Nd = mg_data->vlen[nlevel];
    double *lastA_inv;
    CSR_mat_to_dense_mat(mg_data->A[nlevel], &lastA_inv);
    if (BCx + BCy + BCz == 0)
    {
        // Pure periodic BC, rank deficient, use pseudo inverse
        int m = last_Nd, n = last_Nd;
        int min_mn = (m > n) ? n : m;
        double *U  = (double*) malloc(sizeof(double) * m * m);
        double *S  = (double*) malloc(sizeof(double) * min_mn);
        double *T  = (double*) malloc(sizeof(double) * m * n);
        double *VT = (double*) malloc(sizeof(double) * n * n);
        double *sb = (double*) malloc(sizeof(double) * min_mn);
        assert(U != NULL && VT != NULL && T != NULL && S != NULL && sb != NULL);
        
        int dgesvd_info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'A', 'A', m, n, 
            lastA_inv, n, S, U, m, VT, n, sb
        );
        assert(dgesvd_info == 0);
        // Inverse the diagonal of S
        for (int i = 0; i < min_mn; i++) 
        {
            if (fabs(S[i]) > 1e-15) S[i] = 1.0 / S[i];
            else S[i] = 0.0;
        }
        // invS^T * U^T, == (U * invS)^T
        #pragma omp parallel for
        for (int i = 0; i < m; i++)
        {
            double *U_i = U + i * m;
            double *T_i = T + i * n;
            for (int j = 0; j < min_mn; j++)
                T_i[j] = U_i[j] * S[j];
            for (int j = min_mn; j < n; j++) T_i[j] = 0.0;
        }
        // pinv = V * (invS^T * U^T) == VT^T * (U * invS)^T
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasTrans, m, m, n,
            1.0, VT, n, T, n, 0.0, lastA_inv, last_Nd
        );
        
        free(U);
        free(S);
        free(T);
        free(VT);
        free(sb);
        mg_data->use_pinv   = 1;
        mg_data->lastA_inv  = lastA_inv;
        mg_data->lastA_ipiv = NULL;
    } else {
        // Not pure periodic BC, full rank, use LU decomposition. The data in lastA_inv 
        // and lastA_ipiv can be directly used in LAPACKE_dgetrs later
        int *lastA_ipiv = (int*) malloc(sizeof(int) * last_Nd);
        assert(lastA_ipiv != NULL);
        int dgetrf_info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, last_Nd, last_Nd, lastA_inv, last_Nd, lastA_ipiv);
        assert(dgetrf_info == 0);
        mg_data->use_pinv   = 0;
        mg_data->lastA_inv  = lastA_inv;
        mg_data->lastA_ipiv = lastA_ipiv;
    }  // End of "if (BCx + BCy + BCz == 0)"

    *mg_data_ = mg_data;
}

void MG_solve(mg_data_t mg_data, const double *b, double *x, const double reltol)
{
    int n_0 = mg_data->vlen[0];
    int max_vcycle = 100;
    
    double b_l2_norm = 0.0;
    #pragma omp parallel for simd reduction(+:b_l2_norm)
    for (int i = 0; i < n_0; i++)
        b_l2_norm += b[i] * b[i];
    b_l2_norm = sqrt(b_l2_norm);
    
    // x0 = zeros(size(mg.A{1}, 1), 1);
    // r{1} = b - mg.A{1} * x;
    memset(x, 0, sizeof(double) * n_0);
    double *rv_0 = mg_data->rv[0];
    double *ev_0 = mg_data->ev[0];
    CSR_SpMV(mg_data->A[0], x, rv_0);
    #pragma omp parallel for simd
    for (int i = 0; i < n_0; i++)
        rv_0[i] = b[i] - rv_0[i];
    
    int nlevel = mg_data->nlevel;
    for (int k = 0; k < max_vcycle; k++)
    {
        // 1. Downward pass
        for (int lvl = 0; lvl < nlevel; lvl++)
        {
            int    n_lvl   = mg_data->vlen[lvl];
            double *ev_lvl = mg_data->ev[lvl];
            double *rv_lvl = mg_data->rv[lvl];
            double *tv_lvl = mg_data->tv[lvl];
            double *M_lvl  = mg_data->M[lvl];
            
            // Pre-smoothing
            // e{level} = mg.M{level} .* r{level};
            #pragma omp parallel for simd
            for (int i = 0; i < n_lvl; i++)
                ev_lvl[i] = M_lvl[i] * rv_lvl[i];
            // Restrict the residual
            // t = mg.A{level} * e{level}
            CSR_SpMV(mg_data->A[lvl], ev_lvl, tv_lvl);
            // t = r{level} - t
            #pragma omp parallel for simd
            for (int i = 0; i < n_lvl; i++)
                tv_lvl[i] = rv_lvl[i] - tv_lvl[i];
            // r{level+1} = mg.R{level+1} * t
            CSR_SpMV(mg_data->R[lvl], tv_lvl, mg_data->rv[lvl + 1]);
        }  // End of lvl loop
        
        // 2. Solve on the coarsest level
        int last_Nd = mg_data->vlen[nlevel];
        if (mg_data->use_pinv == 0)
        {
            // mg.e{level+1} = mg.lastA_U \ (mg.lastA_L \ mg.r{mg.nlevel});
            LAPACKE_dgetrs(
                LAPACK_ROW_MAJOR, 'N', last_Nd, 1, 
                mg_data->lastA_inv, last_Nd, mg_data->lastA_ipiv, 
                mg_data->rv[nlevel], 1
            );
            memcpy(mg_data->ev[nlevel], mg_data->rv[nlevel], sizeof(double) * last_Nd);
        } else {
            // mg.e{level+1} = mg.lastA_pinv * mg.r{mg.nlevel};
            cblas_dgemv(
                CblasRowMajor, CblasNoTrans, last_Nd, last_Nd, 1.0, mg_data->lastA_inv, 
                last_Nd, mg_data->rv[nlevel], 1, 0.0, mg_data->ev[nlevel], 1
            );
        }
        
        // 3. Upward pass
        for (int lvl = nlevel - 1; lvl >= 0; lvl--)
        {
            int    n_lvl   = mg_data->vlen[lvl];
            double *ev_lvl = mg_data->ev[lvl];
            double *rv_lvl = mg_data->rv[lvl];
            double *tv_lvl = mg_data->tv[lvl];
            double *M_lvl  = mg_data->M[lvl];
            
            // Prolong the correction
            // e{level} = e{level} + mg.P{level+1} * e{level+1}; 
            CSR_SpMV(mg_data->P[lvl], mg_data->ev[lvl + 1], tv_lvl);
            #pragma omp parallel for simd
            for (int i = 0; i < n_lvl; i++)
                ev_lvl[i] += tv_lvl[i];
            
            // Post-smoothing
            // t = mg.A{level} * e{level}
            CSR_SpMV(mg_data->A[lvl], ev_lvl, tv_lvl);
            // e{level} = e{level} + mg.M{level} .* (r{level} - t)
            #pragma omp parallel for simd
            for (int i = 0; i < n_lvl; i++)
                ev_lvl[i] += M_lvl[i] * (rv_lvl[i] - tv_lvl[i]);
        }  // End of lvl loop
        
        // 4. Correct the finest grid solution, calculate the new residual,
        //    and check the relative error
        // x = x + e{1};
        // r{1} = b - mg.A{1} * x;
        #pragma omp parallel for simd
        for (int i = 0; i < n_0; i++) x[i] += ev_0[i];
        CSR_SpMV(mg_data->A[0], x, rv_0);
        double res_l2_norm = 0.0, res_relerr;
        #pragma omp parallel for simd reduction(+:res_l2_norm)
        for (int i = 0; i < n_0; i++)
        {
            rv_0[i] = b[i] - rv_0[i];
            res_l2_norm += rv_0[i] * rv_0[i];
        }
        res_l2_norm = sqrt(res_l2_norm);
        res_relerr  = res_l2_norm / b_l2_norm;
        printf("%2d    %e\n", k, res_relerr);
        if (res_relerr <= reltol) break;
    }  // End of k loop
}

