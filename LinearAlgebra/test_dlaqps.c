#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <cblas.h>
#include <lapacke.h>

// Not sure if this dgeqp3_reltol works. The accuracy control seems to be buggy.

// Partial pivoting QR factorization, simplified output BLAS-3 version.
// This function is based on LAPACK function dgeqp3(). 
// The partial pivoting QR factorization is of form:
//     A * P = Q * [R11, R12; 0, R22]
// where R11 is an upper-triangular matrix, R12 and R22 are dense matrices,
// P is a permutation matrix. 
// Input parameters:
//   m      : Number of rows of the matrix A
//   n      : Number of columns of the matrix A
//   a      : Size lda * n, target column-major matrix A
//   lda    : Leading dimension of A
//   jpvt   : Size n, matrix A column permutation array, A(:, jpvt) = A * P
//   work   : Size n*3 + (n+1)*max_nb, where max_nb = 8, work buffer
//   reltol : Matrix column norm decrease relative tolerance
// Output parameters:
//   a        : [H11 R12; H21 R22], where the upper-triangular part of H11 = R11
//   <return> : Dimension of upper-triangular matrix R11 (stop rank)
int dgeqp3_reltol(
    const int m, const int n, double *a, const int lda, int *jpvt, 
    double *work, const double reltol
)
{
    extern int dlaqps_(
        const int *m, const int *n, const int *offset, const int *nb, const int *kb, 
        double *a, const int *lda, int *jpvt, double *tau, double *vn1, double *vn2, 
        double *auxv, double *f, const int *ldf
    );

    int    max_nb = 8;
    int    min_mn = (m < n) ? m : n;
    double *vn1   = work;
    double *vn2   = vn1 + n;
    double *auxv  = vn2 + n;
    double *f     = auxv + max_nb;
    double *tau   = f + n * max_nb; 

    // Prepare column norms for dlaqps()
    double max_colnorm = 0.0;
    for (int j = 0; j < n; ++j) 
    {
        double *aj = a + j * lda;
        double col_norm = 0.0;
        for (int i = 0; i < m; i++) col_norm += aj[i] * aj[i];
        vn1[j]  = sqrt(col_norm);
        vn2[j]  = vn1[j];
        jpvt[j] = j;
        if (vn1[j] > max_colnorm) max_colnorm = vn1[j];
    }
    double stop_norm = max_colnorm * reltol;

    // Compute multiple steps of QR factorization with column pivoting of
    // matrix block a(:, j : n) using level 3 BLAS, j = max_nb * {1, 2, 3, ...}
    int j = 0, rank = min_mn, converged = 0;
    while (j < min_mn && converged == 0)
    {
        int nj = n - j;         // Number of columns of the matrix block a(:, j : n)
        int nb = min_mn - j;    // Number of columns to factorize
        int kb;                 // Number of columns actually factorized
        if (max_nb < nb) nb = max_nb;
        dlaqps_(
            &m, &nj, &j, &nb, &kb, &a[j * lda], &lda, &jpvt[j], 
            &tau[j], &vn1[j], &vn2[j], auxv, f, &nj
        );
        j += kb;
        
        double col_norm = 0.0;
        double *ajj = a + j * lda + j;
        for (int i = 0; i < m - j; i++) col_norm += ajj[i] * ajj[i];
        col_norm = sqrt(col_norm);
        if (col_norm <= stop_norm)
        {
            rank = j;
            converged = 1;
        }

        printf("%s, %d: nb = %d, kb = %d, j = %d, col_norm = %e\n", __FUNCTION__, __LINE__, nb, kb, j, col_norm);
    }
    return rank;
}

int main(int argc, char **argv)
{
    int m, n;
    m = atoi(argv[1]);
    n = atoi(argv[2]);

    int nb = 8, offset = 0, kb;
    double *A    = (double*) malloc(sizeof(double) * m * n);
    int    *jpvt = (int*)    malloc(sizeof(int)    * n);

    for (int j = 0; j < n; j++)
    {
        double xj = 5.14 + (double) j;
        double yj = 8.10 - (double) j;
        for (int i = 0; i < m; i++)
        {
            double xi = -1.14 - (double) i;
            double yi = -8.93 + (double) i;
            double dx = xj - xi;
            double dy = yj - yi;
            double Aji = sqrt(dx * dx + dy * dy);
            A[j * m + i] = Aji;
        }
    }

    int lwork = n * 3 + (n + 1) * nb;
    double norm_reltol = atof(argv[3]);
    double *work = (double*) malloc(sizeof(double) * lwork);
    int rank = dgeqp3_reltol(m, n, A, m, jpvt, work, norm_reltol);
    printf("rank = %d\n", rank);

    printf("A on exit:\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++) 
            printf("% 3.4lf  ", A[j * m + i]);
        printf("\n");
    }
    printf("\n");
    printf("jpvt: ");
    for (int j = 0; j < n; j++) printf("%2d ", jpvt[j]+1);
    printf("\n");
    printf("vn1: ");
    for (int j = 0; j < n; j++) printf("% .3e ", work[j]);
    printf("\n");
    printf("vn2: ");
    for (int j = 0; j < n; j++) printf("% .3e ", work[n + j]);
    printf("\n");

    free(A);
    free(jpvt);
    free(work);
}