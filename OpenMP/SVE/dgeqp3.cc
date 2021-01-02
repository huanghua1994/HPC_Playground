// To compile this file:
// icc -O3 -xHost -fopenmp dgeqp3.cc -o dgeqp3.exe -mkl
// gcc -O3 -march=native -fopenmp dgeqp3.cc -o dgeqp3.exe -lm $HOME/softwares/OpenBLAS-0.13/lib/libopenblas.a -lgfortran
// (Cray CC) CC -O3 -h omp dgeqp3.cc -o dgeqp3.exe, but cannot link libpmi.so.0 on octavius yet

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MIN(a, b) ((a) < (b) ? (a) :(b))
#define MAX(a, b) ((a) > (b) ? (a) :(b))

extern "C" void dgeqpf_(int *M, int *N, double *A, int *LDA, int *JPVT, double *TAU, double *WORK, int *INFO);
extern "C" void dgeqp3_(int *M, int *N, double *A, int *LDA, int *JPVT, double *TAU, double *WORK, int *LWORK, int *INFO);

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

    int nb = 128;
    int lwork = 2 * ncol + (ncol + 1) * nb;

    double *A       = (double *) malloc(sizeof(double) * ncol * nrow);
    double *A0      = (double *) malloc(sizeof(double) * ncol * nrow);
    double *QR_buff = (double *) malloc(sizeof(double) * ncol);
    double *tau     = (double *) malloc(sizeof(double) * MIN(ncol, nrow));
    double *work    = (double *) malloc(sizeof(double) * lwork);
    int    *jpvt    = (int *)    malloc(sizeof(int)    * ncol);
    srand48(time(NULL));
    double *x1 = (double*) malloc(sizeof(double) * ncol);
    double *y1 = (double*) malloc(sizeof(double) * ncol);
    double *x2 = (double*) malloc(sizeof(double) * nrow);
    double *y2 = (double*) malloc(sizeof(double) * nrow);
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
        double *A_icol = A + icol * nrow;
        for (int irow = 0; irow < nrow; irow++)
        {
            double dx = x1[icol] - x2[irow];
            double dy = y1[icol] - y2[irow];
            double d  = sqrt(dx * dx + dy * dy);
            A_icol[irow] = 1.0 / d;
        }
    }
    memcpy(A0, A, sizeof(double) * nrow * ncol);
    free(x1);
    free(y1);
    free(x2);
    free(y2);

    int info;
    for (int itest = 0; itest <= ntest; itest++)
    {
        double start_t = omp_get_wtime();
        //dgeqpf_(&nrow, &ncol, A, &nrow, jpvt, tau, work, &info);
        dgeqp3_(&nrow, &ncol, A, &nrow, jpvt, tau, work, &lwork, &info);
        double stop_t  = omp_get_wtime();
        double run_ms  = 1000.0 * (stop_t - start_t);
        if (itest > 0) printf("dgeqp3_ used %.2f ms, info = %d\n", run_ms, info);
        memcpy(A, A0, sizeof(double) * ncol * nrow);
    }

    free(A);
    free(A0);
    free(QR_buff);
}
