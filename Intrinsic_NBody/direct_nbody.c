#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "kernels.h"

void test_direct_nbody(
    const int n_src, const DTYPE *src_coord, DTYPE *src_val, 
    const int n_trg, const DTYPE *trg_coord, DTYPE *trg_val,
    kernel_matvec_fptr krnl_matvec
)
{
    int nthread = omp_get_max_threads();
    for (int k = 0; k < 5; k++)
    {
        double st = omp_get_wtime();
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            int trg_sidx = tid * n_trg / nthread;
            int trg_eidx = (tid + 1) * n_trg / nthread;
            int n_trg_thread = trg_eidx - trg_sidx;
            krnl_matvec(
                trg_coord + trg_sidx, n_trg, n_trg_thread, 
                src_coord, n_src, n_src, 
                src_val, trg_val
            );
        }
        double et = omp_get_wtime();
        printf("Direct N-Body %2d: %.3lf s\n", k, et - st);
    }
}

int main(int argc, char **argv)
{
    int n_src = 0, n_trg = 0;
    if (argc >= 2) n_src = atoi(argv[1]);
    if (argc >= 3) n_trg = atoi(argv[2]);
    if (n_src == 0)
    {
        printf("n_src = ");
        scanf("%d", &n_src);
    }
    if (n_trg == 0)
    {
        printf("n_trg = ");
        scanf("%d", &n_trg);
    }
    
    DTYPE *src_coord = (DTYPE*) malloc(sizeof(DTYPE) * n_src * 3);
    DTYPE *trg_coord = (DTYPE*) malloc(sizeof(DTYPE) * n_trg * 3);
    DTYPE *src_val   = (DTYPE*) malloc(sizeof(DTYPE) * n_src);
    DTYPE *trg_val   = (DTYPE*) malloc(sizeof(DTYPE) * n_src);
    srand48(time(NULL));
    for (int i = 0; i < n_src * 3; i++) src_coord[i] = drand48();
    for (int i = 0; i < n_trg * 3; i++) trg_coord[i] = drand48();
    for (int i = 0; i < n_src; i++) src_val[i] = drand48();
    
    kernel_matvec_fptr krnl_matvec = reciprocal_matvec_ref;
    test_direct_nbody(
        n_src, src_coord, src_val,
        n_trg, trg_coord, trg_val, 
        krnl_matvec
    );
    
    free(src_coord);
    free(trg_coord);
    free(src_val);
    free(trg_val);
    return 0;
}