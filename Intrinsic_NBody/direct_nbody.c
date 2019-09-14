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
    int n_trg_SIMD = n_trg / SIMD_LEN;
    
    for (int k = 0; k < 5; k++)
    {
        double st = omp_get_wtime();
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            int trg_SIMD_sidx = tid * n_trg_SIMD / nthread;
            int trg_SIMD_eidx = (tid + 1) * n_trg_SIMD / nthread;
            
            int trg_sidx = trg_SIMD_sidx * SIMD_LEN;
            int trg_eidx = trg_SIMD_eidx * SIMD_LEN;
            int n_trg_thread = trg_eidx - trg_sidx;
            krnl_matvec(
                trg_coord + trg_sidx, n_trg, n_trg_thread, 
                src_coord, n_src, n_src, 
                src_val, trg_val + trg_sidx
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
    
    int n_src_SIMD = (n_src + SIMD_LEN - 1) / SIMD_LEN * SIMD_LEN;
    int n_trg_SIMD = (n_trg + SIMD_LEN - 1) / SIMD_LEN * SIMD_LEN;
    DTYPE *src_coord = (DTYPE*) _mm_malloc(sizeof(DTYPE) * n_src_SIMD * 3, 64);
    DTYPE *trg_coord = (DTYPE*) _mm_malloc(sizeof(DTYPE) * n_trg_SIMD * 3, 64);
    DTYPE *src_val   = (DTYPE*) _mm_malloc(sizeof(DTYPE) * n_src_SIMD, 64);
    DTYPE *trg_val0  = (DTYPE*) _mm_malloc(sizeof(DTYPE) * n_trg_SIMD, 64);
    DTYPE *trg_val1  = (DTYPE*) _mm_malloc(sizeof(DTYPE) * n_trg_SIMD, 64);
    srand48(time(NULL));
    for (int i = 0; i < n_src; i++) 
    {
        src_coord[i + n_src_SIMD * 0] = drand48();
        src_coord[i + n_src_SIMD * 1] = drand48();
        src_coord[i + n_src_SIMD * 2] = drand48();
        src_val[i] = drand48();
    }
    for (int i = n_src; i < n_src_SIMD; i++)
        {
        src_coord[i + n_src_SIMD * 0] = 0;
        src_coord[i + n_src_SIMD * 1] = 0;
        src_coord[i + n_src_SIMD * 2] = 0;
        src_val[i] = 0;
    }
    for (int i = 0; i < n_trg; i++) 
    {
        trg_coord[i + n_trg_SIMD * 0] = drand48();
        trg_coord[i + n_trg_SIMD * 1] = drand48();
        trg_coord[i + n_trg_SIMD * 2] = drand48();
        trg_val0[i] = 0.0;
        trg_val1[i] = 0.0;
    }
    
    printf("Reference auto-vectorized kernel:\n");
    kernel_matvec_fptr ref_matvec = reciprocal_matvec_ref;
    test_direct_nbody(
        n_src, src_coord, src_val,
        n_trg, trg_coord, trg_val0, 
        ref_matvec
    );
    
    printf("AVX intrinsic kernel:\n");
    kernel_matvec_fptr avx_matvec = reciprocal_matvec_avx_aligned;
    test_direct_nbody(
        n_src, src_coord, src_val,
        n_trg, trg_coord, trg_val1, 
        avx_matvec
    );
    
    DTYPE ref_l2 = 0.0, err_l2 = 0.0;
    for (int i = 0; i < n_trg; i++)
    {
        DTYPE diff = trg_val0[i] - trg_val1[i];
        ref_l2 += trg_val0[i] * trg_val0[i];
        err_l2 += diff * diff;
    }
    ref_l2 = DSQRT(ref_l2);
    err_l2 = DSQRT(err_l2);
    printf("AVX intrinsic kernel result relative L2 error = %e\n", err_l2 / ref_l2);
    
    _mm_free(src_coord);
    _mm_free(trg_coord);
    _mm_free(src_val);
    _mm_free(trg_val0);
    _mm_free(trg_val1);
    return 0;
}