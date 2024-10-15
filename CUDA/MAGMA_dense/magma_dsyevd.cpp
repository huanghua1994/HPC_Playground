#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <omp.h>

#include "magma_v2.h"
#include "magma_lapack.h"

int main(int argc, char **argv)
{
    if (argc < 3) 
    {
        printf("Usage: %s <matrix-size> <n-test>\n", argv[0]);
        exit(1);
    }

    const int m = atoi(argv[1]);
    const int n_test = atoi(argv[2]);
    const int ldA = m;

    // Initialize MAGMA
    magma_init();
    magma_queue_t queue;
    magma_queue_create(0, &queue);

    double *h_A = NULL, *d_A = NULL;
    size_t A_bytes = sizeof(double) * ldA * m;
    magma_malloc_cpu((void **) &h_A, A_bytes);
    magma_malloc((void **) &d_A, A_bytes);
    assert(h_A != NULL);
    assert(d_A != NULL);
    for (int j = 0; j < m; j++)
    {
        double *h_A_j = h_A + j * ldA;
        for (int i = 0; i < m; i++) h_A_j[i] = rand() / (double) RAND_MAX;
    }
    magma_setmatrix(m, m, sizeof(double), h_A, ldA, d_A, ldA, queue);

    double *w = NULL;   // Eigen values
    double *wA = NULL;
    magma_malloc_cpu((void **) &w, sizeof(double) * m);
    magma_malloc_cpu((void **) &wA, A_bytes);
    assert(w != NULL);
    assert(wA != NULL);
    const int ldwA = ldA;

    // Query workspace
    int lwork = -1, liwork = -1, iwork_query = 0, info = 0;
    int *iwork = NULL;
    double work_query = 0;
    double *work = NULL;
    magma_dsyevd_gpu(
        MagmaVec, MagmaLower, m, d_A, ldA, w, wA, ldwA,
        &work_query, lwork, &iwork_query, liwork, &info
    );
    assert(info == 0);
    lwork = (int) work_query;
    liwork = iwork_query;
    printf("Queried lwork = %d, liwork = %d\n", lwork, liwork);
    magma_malloc_cpu((void **) &work, sizeof(double) * lwork);
    magma_malloc_cpu((void **) &iwork, sizeof(int) * liwork);

    // Warm-up running
    magma_dsyevd_gpu(
        MagmaVec, MagmaLower, m, d_A, ldA, w, wA, ldwA,
        work, lwork, iwork, liwork, &info
    );
    assert(info == 0);
    printf("GPU warm-up running finished.\n\n");
    
    // Test running
    for (int i = 0; i < n_test; i++)
    {
        magma_setmatrix(m, m, sizeof(double), h_A, ldA, d_A, ldA, queue);
        // See testing/testing_zpotrf_gpu.cpp for timing methods
        double st = magma_wtime();
        magma_dsyevd_gpu(
            MagmaVec, MagmaLower, m, d_A, ldA, w, wA, ldwA,
            work, lwork, iwork, liwork, &info
        );
        double et = magma_wtime();
        assert(info == 0);
        printf("%.3f ms\n", 1000.0 * (et - st));
    }

    magma_free_cpu(h_A);
    magma_free(d_A);
    magma_free_cpu(w);
    magma_free_cpu(wA);
    magma_free_cpu(work);
    magma_free_cpu(iwork);

    magma_finalize();
    return 0;
}
