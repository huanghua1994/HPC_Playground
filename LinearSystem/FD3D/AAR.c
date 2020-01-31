#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <mkl.h>

#include "FD3D.h"

double L2Norm(const int length, const double *x)
{
    double res = 0.0;
    #pragma omp parallel for simd reduction(+:res)
    for (int i = 0; i < length; i++)
        res += x[i] * x[i];
    res = sqrt(res);
    return res;
}

void calcPreconResVec(const int length, const double *r, double *f)
{
    const double inv_M = -1.0 / (3.0 * stencil_coefs[0]);
    #pragma omp parallel for simd 
    for (int i = 0; i < length; i++) f[i] = inv_M * r[i];
}

void calcResVec(const int N, const double *x, const double *b, double *r)
{
    FD3D_Laplacian_MatVec(x, r);
    #pragma omp parallel for simd 
    for (int i = 0; i < N; i++) r[i] = b[i] - r[i];
}

const int m = 7;
const int p = 6;
const double omega = 0.6;
const double beta  = 0.6;

// Anderson extrapolation update 
// x_{k+1} = x_k + omega * f
// Input parameters:
//    N     : Size of operated vectors
//    xk, f : Input vectors, size N
//    omega : Scaling factor
// Output parameters:
//    xkp1 : Result vector, size N
void RichardsonUpdate(
    const int N, const double *xk, const double *f, 
    const double omega, double *xkp1
)
{
    #pragma omp parallel for simd
    for (int i = 0; i < N; i++)
        xkp1[i] = xk[i] + omega * f[i];
}

// Anderson extrapolation update 
// x_{k+1} = x_k + beta * f - (X + beta * F) * inv(F^T * F) * F^T * f
// Input parameters:
//    N     : Size of operated vectors
//    m     : Number of vectors in X and F
//    xk, f : Input vectors, size N
//    X, F  : Input matrices, size N * m
//    beta  : Scaling factor
// Output parameters:
//    xkp1 : Result vector, size N
double *FTF = NULL;
double *FTf = NULL;
double *s   = NULL;
double *XF  = NULL;
double *tmp = NULL;
void AndersonExtrapolation(
    const int N, const int m, const double *xk, const double *f,
    const double *X, const double *F, const double beta, double *xkp1
)
{
    // (1) Compute FTF := F^T * F
    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans, m, m, N,
        1.0, F, N, F, N, 0.0, FTF, m
    );
    
    // (2) Compute FTf := F^T * f
    cblas_dgemv(
        CblasColMajor, CblasTrans, N, m, 
        1.0, F, N, f, 1, 0.0, FTf, 1
    );
    
    // (3) Compute inv(F^T * F) * (F^T * f) by solving (F^T * F) * x = F^T * f
    // LAPACKE_dgelsd() uses SVD to get the minimum residual solution and stores the result in FTf
    int FTF_rank;
    LAPACKE_dgelsd(
        LAPACK_COL_MAJOR, m, m, 1, FTF, m, 
        FTf, m, s, -1.0, &FTF_rank
    );
    
    // (4) Compute XF := beta * F + X
    #pragma omp parallel for simd
    for (int i = 0; i < N * m; i++)
        XF[i] = X[i] + beta * F[i];
    
    // (5) Compute (X + beta * F) * inv(F^T * F) * F^T * f, store result in tmp
    cblas_dgemv(
        CblasColMajor, CblasNoTrans, N, m, 
        1.0, XF, N, FTf, 1, 0.0, tmp, 1
    );
    
    // (6) Compute final result
    #pragma omp parallel for simd
    for (int i = 0; i < N; i++)
        xkp1[i] = xk[i] + beta * f[i] - tmp[i];
}

double *r  = NULL;
double *x0 = NULL;
double *f  = NULL;
double *f0 = NULL;
double *X  = NULL;
double *F  = NULL;
int N_old  = 0;
void AAR(
    const int N, const double res_tol, const int max_iter,
    const double *b, double *x
)
{
    if (N > N_old)
    {
        N_old = N;
        free(r);
        free(x0);
        free(f);
        free(f0);
        free(X);
        free(F);
        free(FTF);
        free(FTf);
        free(s);
        free(XF);
        free(tmp);
        size_t vec_msize = sizeof(double) * N;
        size_t m_msize   = sizeof(double) * m;
        r   = (double*) malloc(vec_msize);
        x0  = (double*) malloc(vec_msize);
        f   = (double*) malloc(vec_msize);
        f0  = (double*) malloc(vec_msize);
        X   = (double*) malloc(vec_msize * m);
        F   = (double*) malloc(vec_msize * m);
        FTF = (double*) malloc(m_msize * m);
        FTf = (double*) malloc(m_msize);
        s   = (double*) malloc(m_msize);
        XF  = (double*) malloc(vec_msize * m);
        tmp = (double*) malloc(vec_msize);
        assert(r != NULL && x0 != NULL && f != NULL && f0 != NULL);
        assert(X != NULL && F != NULL && FTF != NULL && FTf != NULL);
        assert(s != NULL && XF != NULL && tmp != NULL);
        memset(X, 0, vec_msize * m);
        memset(F, 0, vec_msize * m);
    }
    
    double b_l2norm, r_l2norm, stop_res;
    double st = omp_get_wtime();
    
    // Initialize x0 as x (initial guess)
    #pragma omp parallel for simd 
    for (int i = 0; i < N; i++) x0[i] = x[i];
    
    // Compute initial residual vector
    calcResVec(N, x, b, r);
    b_l2norm = L2Norm(N, b);
    r_l2norm = L2Norm(N, r);
    stop_res = b_l2norm * res_tol;
    
    int iter = 1;
    //printf("Iteration, Relative Residual\n");
    for (; iter <= max_iter; iter++)
    {
        // (1) Compute preconditioned residual f
        calcPreconResVec(N, r, f);
        
        // (2) Store residual & iteration history
        if (iter > 1)
        {
            int pos = (iter - 2) % m;
            double *Xp = X + pos * N;
            double *Fp = F + pos * N;
            #pragma omp parallel for simd
            for (int i = 0; i < N; i++)
            {
                Xp[i] = x[i] - x0[i];
                Fp[i] = f[i] - f0[i];
            }
        }
        #pragma omp parallel for simd
        for (int i = 0; i < N; i++)
        {
            x0[i] = x[i];
            f0[i] = f[i];
        }
        
        // (3) Update vector
        if ((iter % p == 0) && (iter > 1))
        {
            AndersonExtrapolation(N, m, x0, f, X, F, beta, x);
            calcResVec(N, x, b, r);
            r_l2norm = L2Norm(N, r);
        } else {
            RichardsonUpdate(N, x0, f, omega, x);
            calcResVec(N, x, b, r);
        }
        
        //printf("      %3d,         %e\n", iter, r_l2norm / b_l2norm);
        if (r_l2norm < stop_res) 
        {
            //double et = omp_get_wtime();
            //printf("AAR converged, used time = %lf (s)\n", et - st);
            break;
        }
    }
    
    double et = omp_get_wtime();
    printf(
        "AAR stopped after %d iterations, residual relerr = %e, used %.3lf seconds\n", 
        iter, r_l2norm / b_l2norm, et - st
    );
}

/*
void  AndersonExtrapolation(int N, int m, double *x_kp1, double *x_k, double *f, double *X, double *F, double beta, double *t_mpi) 
{
#define FtF_Ftf(i,j) FtF_Ftf[(j)*m+(i)]
    int i, j, matrank;
    double *FtF_Ftf, *s;
    
    /////////////////
    double t1, t2;
    /////////////////
    
    // Note: the array FtF_Ftf stores  F^T * F and F^T * f
    // |---------------------------------------|------------|
    // |               F^T * F                 |    F^T * f |
    // |---------------------------------------|------------|
    // |<--              m * m              -->|<--   m  -->|
    
    FtF_Ftf = (double *)malloc( (m*m+m) * sizeof(double) );
    s = (double *)malloc( m * sizeof(double) );
    if (FtF_Ftf == NULL || s == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }  

    *t_mpi = 0.0;
    t1 = omp_get_wtime(); 
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, N, 1.0, F, N, F, N, 0.0, FtF_Ftf, m);
    t2 = omp_get_wtime();
    *t_mpi = t2 - t1; 

    // calculate -F^T * f using CBLAS  (LOCAL)
    cblas_dgemv(CblasColMajor, CblasTrans, N, m, -1.0, F, N, f, 1, 0.0, &FtF_Ftf[m*m], 1);
    
    // find inv(F^T * F) * (F^T * f) by solving (F^T * F) * x = F^T * f (LOCAL)
    LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, FtF_Ftf, m, &FtF_Ftf[m*m], m, s, -1.0, &matrank);

    // find beta * F + X (LOCAL)
    double *XF = (double *)malloc( (N*m) * sizeof(double) );
    double *dp = (double *)malloc( N * sizeof(double) );
    for (i = 0; i < N*m; i++) 
        XF[i] = X[i] + beta * F[i];

    // now that XF = beta * F + X, FtF_Ftf+m*m = inv(F^T * F) * (F^T * f), we calculate the last term
    // of the Anderson update by multiplying them together
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, m, 1.0, XF, N, &FtF_Ftf[m*m], 1, 0.0, dp, 1);
    
    free(XF);

    // add beta * f to x_{k+1}
    for (i = 0; i < N; i++)
        x_kp1[i] = x_k[i] + beta * f[i] + dp[i];
    
    free(s);
    free(FtF_Ftf);
    free(dp);
#undef FtF_Ftf 
}

void AAR(const int N, const double tol, const int max_iter, const double *b, double *x)
{
#define X(i,j) X[(j)*N+(i)]
#define F(i,j) F[(j)*N+(i)]
    
    int rank = 0, i, iter_count, i_hist;
    double *r, *x_old, *f, *f_old, *X, *F, b_2norm, r_2norm, res;
    
    
    /////////////////////////////////////////////////
    double t1, t2, tt1, tt2, ttot, t_anderson, ttot2, ttot3, ttot4;
    /////////////////////////////////////////////////
    
    ttot = 0.0;
    ttot2 = 0.0;
    ttot3 = 0.0;
    ttot4 = 0.0;
    
    // allocate memory for storing x, residual and preconditioned residual in the local domain
    r = (double *)malloc( N * sizeof(double) );  // residual vector, r = b - Ax
    x_old   = (double *)malloc( N * sizeof(double) );
    f       = (double *)malloc( N * sizeof(double) ); // preconditioned residual vector, f = inv(M) * r
    f_old   = (double *)malloc( N * sizeof(double) );    
    if (r == NULL || x_old == NULL || f == NULL || f_old == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    // allocate memory for storing X, F history matrices
    X = (double *)calloc( N * m, sizeof(double) ); 
    F = (double *)calloc( N * m, sizeof(double) ); 
    if (X == NULL || F == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // initialize x_old as x0 (initial guess vector)
    for (i = 0; i < N; i++) x_old[i] = x[i];


    t1 = omp_get_wtime();
    b_2norm = L2Norm(N, b);
    t2 = omp_get_wtime();
#ifdef DEBUG
    if (rank < 1) printf("2-norm of RHS = %.13f, which took %.3f ms\n", b_2norm, (t2-t1)*1e3);
#endif
    // TODO: find initial residual vector r = b - Ax, and its 2-norm
    //res_fun(N, x, pSPARC, b, r, comm, &t_anderson);
    calcResVec(N, x, b, r);

    // replace the abs tol by scaled tol: tol * ||b||
    double stop_tol = tol * b_2norm; 
    r_2norm = tol + 1.0; // to reduce communication
    iter_count = 0;
    while (r_2norm > stop_tol && iter_count < max_iter) 
    {
        if (rank == 0) printf("iteration #%d, r_2norm = %.3e\n", iter_count, r_2norm / b_2norm);
        
        // *** calculate preconditioned residual f *** //
        //precond_fun(N, r, pSPARC, f, comm); // f = inv(M) * r
        calcPreconResVec(N, r, f);
        
        // *** store residual & iteration history *** //
        if (iter_count > 0) 
        {
            i_hist = (iter_count - 1) % m;
            for (i = 0; i < N; i++) {
                X(i, i_hist) = x[i] - x_old[i];
                F(i, i_hist) = f[i] - f_old[i];
            }
        }
        
        for (i = 0; i < N; i++) 
        {
            x_old[i] = x[i];
            f_old[i] = f[i];
        }
        
        if((iter_count+1) % p == 0 && iter_count > 0) 
        {
            //Anderson extrapolation update  *
            tt1 = omp_get_wtime();
            AndersonExtrapolation(N, m, x, x, f, X, F, beta, &t_anderson);
            tt2 = omp_get_wtime();
            ttot += (tt2-tt1);
            ttot4 += t_anderson;

            tt1 = omp_get_wtime();
            // update residual r = b - Ax
            //res_fun(N, x, pSPARC, b, r, comm, &t_anderson);
            calcResVec(N, x, b, r);
            tt2 = omp_get_wtime();
            ttot2 += (tt2 - tt1);
            ttot3 += t_anderson;
            //Vector2Norm(r, N, &r_2norm, comm); // r_2norm = ||r||
            r_2norm = L2Norm(N, r);
        } else {
            //  Richardson update
            for (i = 0; i < N; i++)
                x[i] = x[i] + omega * f[i];  
            tt1 = omp_get_wtime();   
            // update residual r = b - Ax
            //res_fun(N, x, pSPARC, b, r, comm, &t_anderson);  
            calcResVec(N, x, b, r);
            tt2 = omp_get_wtime();
            ttot2 += (tt2 - tt1);
            ttot3 += t_anderson;
        }
        iter_count++;
        //if (iter_count % 250 == 0) MPI_Barrier(comm);
    }

    // deallocate memory
    free(x_old);
    free(f_old);
    free(f);
    free(r);
    free(X);
    free(F);
#undef X
#undef F
}
*/