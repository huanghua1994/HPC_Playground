#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <cblas.h>
#include <lapacke.h>

int main(int argc, char **argv)
{
    int m, n;
    m = atoi(argv[1]);
    n = atoi(argv[2]);

    int info;
    int    *jpvt = (int*)    malloc(sizeof(int)    * n);
    double *A    = (double*) malloc(sizeof(double) * m * n);
    double *tau  = (double*) malloc(sizeof(double) * n);
    
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

    for (int j = 0; j < n; j++) jpvt[j] = 0;
    int lwork = -1;
    double work_query;
    dgeqp3_(&m, &n, A, &m, jpvt, tau, &work_query, &lwork, &info);
    lwork = (int) work_query;
    double *work = (double*) malloc(sizeof(double) * lwork);
    dgeqp3_(&m, &n, A, &m, jpvt, tau, work, &lwork, &info);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++) 
            printf("% 3.4lf  ", A[j * m + i]);
        printf("\n");
    }
    printf("\n");
    for (int j = 0; j < n; j++) printf("%2d ", jpvt[j]);
    printf("\n");
    for (int j = 0; j < n; j++) printf("% .3e ", work[j]);
    printf("\n");

    free(jpvt);
    free(A);
    free(tau);
    free(work);
}