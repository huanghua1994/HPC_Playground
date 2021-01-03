#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define DTYPE double
#define DABS  fabs
#define DSQRT sqrt
#define MIN(a, b) ((a) < (b) ? (a) :(b))
#define MAX(a, b) ((a) > (b) ? (a) :(b))

static inline void swap_int(int *x, int *y, int len)
{
    int tmp;
    for (int i = 0; i < len; i++)
    {
        tmp  = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
}

static inline void swap_DTYPE(DTYPE *x, DTYPE *y, int len)
{
    DTYPE tmp;
    for (int i = 0; i < len; i++)
    {
        tmp  = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
}

static DTYPE CBLAS_NRM2(const int n, const DTYPE *x, const int incx)
{
    DTYPE nrm2 = 0.0;
    #ifdef SAFE_NRM2
    DTYPE scale = 0.0, ssq = 1.0, absxi, tmp;
    for (int ix = 0; ix < n * incx; ix += incx)
    {
        if (x[ix] == 0.0) continue;
        absxi = DABS(x[ix]);
        if (scale < absxi)
        {
            tmp = scale / absxi;
            ssq = 1.0 + ssq * tmp * tmp;
            scale = absxi;
        } else {
            tmp = absxi / scale;
            ssq = ssq + tmp * tmp;
        }
    }
    nrm2 = scale * DSQRT(ssq);
    #else
    if (incx > 1)
    {
        #pragma omp simd
        for (int ix = 0; ix < n * incx; ix += incx) nrm2 += x[ix] * x[ix];    
    } else {
        #pragma omp simd
        for (int i = 0; i < n; i++) nrm2 += x[i] * x[i];    
    }
    nrm2 = DSQRT(nrm2);
    #endif
    return nrm2;
}

static DTYPE CBLAS_DOT(const int n, const DTYPE *x, const int incx, const DTYPE *y, const int incy)
{
    DTYPE dot = 0.0;
    if (incx > 1 || incy > 1)
    {
        #pragma omp simd
        for (int i = 0; i < n; i++) dot += x[i * incx] * y[i * incy];
    } else {
        #pragma omp simd
        for (int i = 0; i < n; i++) dot += x[i] * y[i];
    }
    return dot;
}

// This function is adapted from H2Pack/H2Pack_ID_compress.c
// Partial pivoting QR decomposition, simplified output version
// The partial pivoting QR decomposition is of form:
//     A * P = Q * [R11, R12; 0, R22]
// where R11 is an upper-triangular matrix, R12 and R22 are dense matrices,
// P is a permutation matrix. 
// Input parameters:
//   A           : Size >= nrow * ldA, target matrix, stored in column major
//   n{row, col} : Number of rows and columns of A
//   ldA         : Leading dimension of A
//   tol_rank    : QR stopping parameter, maximum column rank, 
//   tol_norm    : QR stopping parameter, maximum column 2-norm
//   rel_norm    : If tol_norm is relative to the largest column 2-norm in A
//   n_thread    : Number of threads used in this function
//   QR_buff     : Size ncol, working buffer for partial pivoting QR
// Output parameters:
//   A : Matrix R: [R11, R12; 0, R22]
//   p : Size ncol, matrix A column permutation array, A(:, p) = A * P
//   r : Dimension of upper-triangular matrix R11
void H2P_partial_pivot_QR(
    DTYPE *A, const int nrow, const int ncol, const int ldA, 
    const int tol_rank, const DTYPE tol_norm, const int rel_norm, 
    int *p, int *r, const int n_thread, DTYPE *QR_buff
)
{
    DTYPE *R = A;
    int ldR  = ldA;
    int max_iter = MIN(nrow, ncol);
    
    DTYPE *col_norm = QR_buff; 
    
    // Find a column with largest 2-norm
    #pragma omp parallel for if (n_thread > 1) \
    num_threads(n_thread) schedule(static)
    for (int j = 0; j < ncol; j++)
    {
        p[j] = j;
        col_norm[j] = CBLAS_NRM2(nrow, R + j * ldR, 1);
    }
    DTYPE norm_p = 0.0;
    int pivot = 0;
    for (int j = 0; j < ncol; j++)
    {
        if (col_norm[j] > norm_p)
        {
            norm_p = col_norm[j];
            pivot = j;
        }
    }
    
    // Scale the stopping norm
    int stop_rank = MIN(max_iter, tol_rank);
    DTYPE norm_eps = DSQRT((DTYPE) nrow) * 1e-15;
    DTYPE stop_norm = MAX(norm_eps, tol_norm);
    if (rel_norm) stop_norm *= norm_p;
    
    int rank = -1;
    // Main iteration of Household QR
    for (int i = 0; i < max_iter; i++)
    {
        // 1. Check the stop criteria
        if ((norm_p < stop_norm) || (i >= stop_rank))
        {
            rank = i;
            break;
        }
        
        // 2. Swap the column
        if (i != pivot)
        {
            swap_int(p + i, p + pivot, 1);
            swap_DTYPE(col_norm + i, col_norm + pivot, 1);
            swap_DTYPE(R + i * ldR, R + pivot * ldR, nrow);
        }
        
        // 3. Calculate Householder vector
        int h_len    = nrow - i;
        int h_len_m1 = h_len - 1;
        DTYPE *h_vec = R + i * ldR + i;
        DTYPE sign   = (h_vec[0] > 0.0) ? 1.0 : -1.0;
        DTYPE h_norm = CBLAS_NRM2(h_len, h_vec, 1);
        h_vec[0] = h_vec[0] + sign * h_norm;
        DTYPE inv_h_norm = 1.0 / CBLAS_NRM2(h_len, h_vec, 1);
        #pragma omp simd
        for (int j = 0; j < h_len; j++) h_vec[j] *= inv_h_norm;
        
        // 4. & 5. Householder update & column norm update
        DTYPE *R_block = R + (i + 1) * ldR + i;
        int R_block_nrow = h_len;
        int R_block_ncol = ncol - i - 1;
        #pragma omp parallel for if (n_thread > 1) \
        num_threads(n_thread) schedule(guided)
        for (int j = 0; j < R_block_ncol; j++)
        {
            int ji1 = j + i + 1;
            
            DTYPE *R_block_j = R_block + j * ldR;
            DTYPE h_Rj = 2.0 * CBLAS_DOT(R_block_nrow, h_vec, 1, R_block_j, 1);
            
            // 4. Orthogonalize columns right to the i-th column
            #pragma omp simd
            for (int k = 0; k < R_block_nrow; k++)
                R_block_j[k] -= h_Rj * h_vec[k];
            
            // 5. Update i-th column's 2-norm
            if (col_norm[ji1] < stop_norm)
            {
                col_norm[ji1] = 0.0;
                continue;
            }
            DTYPE tmp = R_block_j[0] * R_block_j[0];
            tmp = col_norm[ji1] * col_norm[ji1] - tmp;
            if (tmp <= 1e-10)
            {
                col_norm[ji1] = CBLAS_NRM2(h_len_m1, R_block_j + 1, 1);
            } else {
                // Fast update 2-norm when the new column norm is not so small
                col_norm[ji1] = DSQRT(tmp);
            }
        }
        
        // We don't need h_vec anymore, can overwrite the i-th column of R
        h_vec[0] = -sign * h_norm;
        memset(h_vec + 1, 0, sizeof(DTYPE) * (h_len - 1));
        // Find next pivot 
        pivot  = i + 1;
        norm_p = 0.0;
        for (int j = i + 1; j < ncol; j++)
        {
            if (col_norm[j] > norm_p)
            {
                norm_p = col_norm[j];
                pivot  = j;
            }
        }
    }
    if (rank == -1) rank = max_iter;
    
    *r = rank;
}

int main(int argc, char **argv)
{
    int nrow, ncol, ntest;
    if (argc < 4)
    {
        printf("Usage: %s nrow ncol ntest\n", argv[0]);
        return 255;
    }
    nrow  = atoi(argv[1]);
    ncol  = atoi(argv[2]);
    ntest = atoi(argv[3]);

    DTYPE *A       = (DTYPE *) malloc(sizeof(DTYPE) * ncol * nrow);
    DTYPE *A0      = (DTYPE *) malloc(sizeof(DTYPE) * ncol * nrow);
    DTYPE *QR_buff = (DTYPE *) malloc(sizeof(DTYPE) * ncol);
    int   *p       = (int *)   malloc(sizeof(int)   * ncol);
    srand48(time(NULL));
    DTYPE *x1 = (DTYPE*) malloc(sizeof(DTYPE) * ncol);
    DTYPE *y1 = (DTYPE*) malloc(sizeof(DTYPE) * ncol);
    DTYPE *x2 = (DTYPE*) malloc(sizeof(DTYPE) * nrow);
    DTYPE *y2 = (DTYPE*) malloc(sizeof(DTYPE) * nrow);
    for (int i = 0; i < ncol; i++) 
    {
        x1[i] = drand48();
        y1[i] = drand48();
    }
    for (int i = 0; i < nrow; i++) 
    {
        x2[i] = drand48() + 0.6;
        y2[i] = drand48() + 0.4;
    }
    for (int icol = 0; icol < ncol; icol++)
    {
        DTYPE *A_icol = A + icol * nrow;
        for (int irow = 0; irow < nrow; irow++)
        {
            DTYPE dx = x1[icol] - x2[irow];
            DTYPE dy = y1[icol] - y2[irow];
            DTYPE d  = sqrt(dx * dx + dy * dy);
            A_icol[irow] = 1.0 / d;
        }
    }
    memcpy(A0, A, sizeof(DTYPE) * nrow * ncol);
    free(x1);
    free(y1);
    free(x2);
    free(y2);

    int rank;
    int nthread = omp_get_max_threads();
    for (int itest = 0; itest <= ntest; itest++)
    {
        double start_t = omp_get_wtime();
        H2P_partial_pivot_QR(
            A, nrow, ncol, nrow, 
            ncol, 1e-8, 1, 
            p, &rank, nthread, QR_buff
        );
        double stop_t  = omp_get_wtime();
        double run_ms  = 1000.0 * (stop_t - start_t);
        if (itest > 0) printf("Partial pivoted QR with 1e-8 reltol used %.2f ms, rank = %d\n", run_ms, rank);
        memcpy(A, A0, sizeof(DTYPE) * ncol * nrow);
    }

    free(A);
    free(A0);
    free(QR_buff);
    free(p);
}