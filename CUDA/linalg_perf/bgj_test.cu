#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_utils.cuh"
#include "cublas_cusolver_init.cuh"
#include <curand.h>

__global__ void set_diag_element_kernel(const int ncol, double *A, const int ldA, const double val)
{
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) A[i * ldA + i] = val;
    __syncthreads();
}

void bgj(
    const int n, double *A, const int ldA, double *B, const int ldB,
    const int blk_size, double *workbuf, const int lwork_getrf, int *ipiv,
    cudaEvent_t *bgj_events, float *bgj_timers
)
{
    float timer_tmp;
    int retval;
    double *getrf_work = workbuf;
    double *diag_blk = workbuf + lwork_getrf;
    memset(bgj_timers, 0, sizeof(float) * 4);
    for (int s_col = 0; s_col < n; s_col += blk_size)
    {
        int curr_bs = blk_size < (n - s_col) ? blk_size : (n - s_col);
        int e_col = s_col + curr_bs;
        int s_row = s_col, e_row = e_col;
        int ncol1 = s_col, ncol2 = n - e_col;
        double *A_diag = A + s_col * ldA + s_row;
        double *B_diag = B + s_col * ldB + s_row;
        
        // e_col = s_col + curr_bs - 1;
        // k_idx = s_col : e_col;
        // idx1  = 1 : s_col-1;
        // idx2  = e_col+1 : n;

        // 1. Inverse current diagonal block
        // [L, U, P] = lu(A(k_idx, k_idx));
        // B(k_idx, k_idx) = inv(A(k_idx, k_idx));
        CUDA_RT_CHECK( cudaEventRecord(bgj_events[0], cublas_stream) );
        CUSOLVER_CHECK( cusolverDnDgetrf(
            cusolver_dn_handle, curr_bs, curr_bs, 
            A_diag, ldA, getrf_work, ipiv, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemset(diag_blk, 0, sizeof(double) * curr_bs * curr_bs) );
        set_diag_element_kernel<<<1, 512>>>(curr_bs, diag_blk, curr_bs, 1.0);
        CUDA_RT_CHECK( cudaPeekAtLastError() );
        CUSOLVER_CHECK( cusolverDnDgetrs(
            cusolver_dn_handle, CUBLAS_OP_N, curr_bs, curr_bs, 
            A_diag, ldA, ipiv, diag_blk, curr_bs, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpy2DAsync(
            B_diag, sizeof(double) * ldB, diag_blk, sizeof(double) * curr_bs, 
            sizeof(double) * curr_bs, curr_bs, cudaMemcpyDeviceToDevice, cusolver_stream
        ) );
        CUDA_RT_CHECK( cudaEventRecord(bgj_events[1], cublas_stream) );

        // 2. Update row panel
        // A(k_idx, idx2) = B(k_idx, k_idx) * A(k_idx, idx2);
        // B(k_idx, idx1) = B(k_idx, k_idx) * B(k_idx, idx1);
        CUSOLVER_CHECK( cusolverDnDgetrs(
            cusolver_dn_handle, CUBLAS_OP_N, curr_bs, ncol2, 
            A_diag, ldA, ipiv, A_diag + curr_bs * ldA, ldA, cusolver_retval
        ) );
        CUSOLVER_CHECK( cusolverDnDgetrs(
            cusolver_dn_handle, CUBLAS_OP_N, curr_bs, ncol1, 
            A_diag, ldA, ipiv, B + s_row, ldB, cusolver_retval
        ) );
        CUDA_RT_CHECK( cudaMemcpyAsync(
            &retval, cusolver_retval, sizeof(int), 
            cudaMemcpyDeviceToHost, cublas_stream
        ) );
        CUDA_RT_CHECK( cudaEventRecord(bgj_events[2], cublas_stream) );

        // 3. Update column panel
        // B(idx1, k_idx) = -A(idx1, k_idx) * B(k_idx, k_idx);
        // B(idx2, k_idx) = -A(idx2, k_idx) * B(k_idx, k_idx);
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            ncol1, curr_bs, curr_bs, 
            &d_neg_one, A + s_col * ldA, ldA, diag_blk, curr_bs, 
            &d_zero, B + s_col * ldB, ldB
        ) );
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            ncol2, curr_bs, curr_bs, 
            &d_neg_one, A_diag + curr_bs, ldA, diag_blk, curr_bs, 
            &d_zero, B_diag + curr_bs, ldB
        ) );
        
        CUDA_RT_CHECK( cudaEventRecord(bgj_events[3], cublas_stream) );

        // 4. Update rest blocks
        // A(idx1, idx2) = A(idx1, idx2) - A(idx1, k_idx) * A(k_idx, idx2);
        // A(idx2, idx2) = A(idx2, idx2) - A(idx2, k_idx) * A(k_idx, idx2);
        // B(idx1, idx1) = B(idx1, idx1) - A(idx1, k_idx) * B(k_idx, idx1);
        // B(idx2, idx1) = B(idx2, idx1) - A(idx2, k_idx) * B(k_idx, idx1);
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            ncol1, ncol2, curr_bs, 
            &d_neg_one, A + s_col * ldA, ldA, A_diag + curr_bs * ldA, ldA, 
            &d_one, A + e_col * ldA, ldA
        ) );
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            ncol2, ncol2, curr_bs, 
            &d_neg_one, A_diag + curr_bs, ldA, A_diag + curr_bs * ldA, ldA, 
            &d_one, A + e_col * ldA + e_row, ldA
        ) );
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            ncol1, ncol1, curr_bs, 
            &d_neg_one, A + s_col * ldA, ldA, B + s_row, ldB, 
            &d_one, B, ldB
        ) );
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            ncol2, ncol1, curr_bs, 
            &d_neg_one, A_diag + curr_bs, ldA, B + s_row, ldB, 
            &d_one, B + e_row, ldB
        ) );
        CUDA_RT_CHECK( cudaEventRecord(bgj_events[4], cublas_stream) );

        // 5. Update timers
        for (int i = 0; i < 5; i++) 
            CUDA_RT_CHECK( cudaEventSynchronize(bgj_events[i]) );
        if (retval != 0)
            fprintf(stderr, "[ERROR] %s, %d: cusolverDnDgetrs returned %d\n", __FILE__, __LINE__, retval);
        CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, bgj_events[0], bgj_events[1]) );
        bgj_timers[0] += timer_tmp;
        CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, bgj_events[1], bgj_events[2]) );
        bgj_timers[1] += timer_tmp;
        CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, bgj_events[2], bgj_events[3]) );
        bgj_timers[2] += timer_tmp;
        CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, bgj_events[3], bgj_events[4]) );
        bgj_timers[3] += timer_tmp;
    }  // End of s_col loop
}

void test_bgj(const int m, const int n_test)
{
    init_cublas_cusolver();

    int ldA = m, lwork_getrf;
    double *A = NULL, *A0 = NULL, *B = NULL, *getrf_work = NULL;
    int *ipiv;

    int blk_size = 512;
    char *bgj_bs_p = getenv("BGJ_BS");
    if (bgj_bs_p != NULL) blk_size = atoi(bgj_bs_p);
    if ((blk_size < 32) || (blk_size > 2048)) blk_size = 512;

    // Allocate memory on device
    CUDA_RT_CHECK( cudaMalloc((void **) &A,  sizeof(double) * m * m) );
    CUDA_RT_CHECK( cudaMalloc((void **) &A0, sizeof(double) * m * m) );
    CUDA_RT_CHECK( cudaMalloc((void **) &B,  sizeof(double) * m * m) );
    CUDA_RT_CHECK( cudaMalloc((void **) &ipiv, sizeof(int) * blk_size) );
    CUSOLVER_CHECK( cusolverDnDgetrf_bufferSize(cusolver_dn_handle, m, m, A, ldA, &lwork_getrf) );
    CUDA_RT_CHECK( cudaMalloc((void **) &getrf_work, sizeof(double) * (lwork_getrf + blk_size * blk_size)) );

    // Initialize random SPD A
    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    CURAND_CHECK( curandGenerateUniformDouble(gen, A0, m * m) );
    CURAND_CHECK( curandDestroyGenerator(gen) );

    // Test dgeqrf and dorgqr
    cudaEvent_t *bgj_events = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * 5);
    for (int i = 0; i < 5; i++)
        CUDA_RT_CHECK( cudaEventCreate(&bgj_events[i]) );
    float *bgj_timers = (float *) malloc(sizeof(float) * 4);
    fprintf(stderr, "diag_inv_ms  row_panel_ms  col_panel_ms  gemm_ms  |  bgj_ms, bgj_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_RT_CHECK( cudaMemcpy(A, A0, sizeof(double) * m * m, cudaMemcpyDeviceToDevice) );
        
        bgj(m, A, ldA, B, ldA, blk_size, getrf_work, lwork_getrf, ipiv, bgj_events, bgj_timers);

        float bgj_ms = 0.0;
        float bgj_mflops = 2.0f * (float) m * (float) m * (float) m / 1000000.0f;
        for (int i = 0; i < 4; i++)
        {
            bgj_ms += bgj_timers[i];
            fprintf(stderr, "%.2f  ", bgj_timers[i]);
        }
        fprintf(stderr, "|  %.2f, %.2f\n", bgj_ms, bgj_mflops / bgj_ms);
    }
    
    // Accuracy check
    double diff_fnorm;
    CUDA_RT_CHECK( cudaMemcpy(A, A0, sizeof(double) * m * m, cudaMemcpyDeviceToDevice) );
    CUDA_RT_CHECK( cudaMemset(A0, 0, sizeof(double) * m * m) );
    set_diag_element_kernel<<<1, 1024>>>(m, A0, m, 1.0);
    CUBLAS_CHECK( cublasDgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, 
        &d_neg_one, A, ldA, B, ldA, &d_one, A0, ldA
    ) );
    CUBLAS_CHECK( cublasDnrm2(cublas_handle, m * m, A0, 1, &diff_fnorm) );
    fprintf(stderr, "BGJ: ||A * inv(A) - I||_{fro} / ||I||_{fro} = %.2e\n", diff_fnorm / sqrt(m));

    CUDA_RT_CHECK( cudaFree(A) );
    CUDA_RT_CHECK( cudaFree(A0) );
    CUDA_RT_CHECK( cudaFree(B) );
    CUDA_RT_CHECK( cudaFree(ipiv) );
    CUDA_RT_CHECK( cudaFree(getrf_work) );
    for (int i = 0; i < 5; i++) CUDA_RT_CHECK( cudaEventDestroy(bgj_events[i]) );
    free(bgj_events);
    free(bgj_timers);
}

int main(int argc, char **argv)
{
    int m = 0, n_test = 10;
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s m n_test \n", argv[0]);
        return 255;
    }
    m = atoi(argv[1]);
    n_test = atoi(argv[2]);
    fprintf(stderr, "m = %d, n_test = %d\n", m, n_test);

    test_bgj(m, n_test);

    return 0;
}