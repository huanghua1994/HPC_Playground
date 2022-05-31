#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_utils.cuh"
#include "cublas_cusolver_init.cuh"
#include <curand.h>

__global__ void gather_P_rows(
    double *A, const int ldA, double *P, const int ldP,
    const int *rowidx, const int curr_bs
)
{
    double *src_ptr = A + rowidx[blockIdx.x];
    double *dst_ptr = P + blockIdx.x;
    for (int i = threadIdx.x; i < curr_bs; i += blockDim.x)
        dst_ptr[i * ldP] = src_ptr[i * ldA];
    __syncthreads();
}

void MGS(
    const int m, const int n, const int blk_size, double *A, const int ldA,
    int *P_rowidx_h, int *P_rowidx_d, int *flags,
    double *workbuf, const int lwork_qr, const int lwork_chol, 
    cudaEvent_t *mgs_events, float *mgs_timers, const int use_xr
)
{
    float timer_tmp;
    int retval;

    // Set up work buffers
    int ldP = 2 * blk_size, ldG = blk_size, ldH = blk_size;
    double *P = workbuf, *G = workbuf, *H = workbuf;
    double *tau = P + 2 * blk_size * blk_size;
    double *work_qr = tau + blk_size;
    double *work_chol = G + blk_size * blk_size;
    for (int s_col = 0; s_col < n; s_col += blk_size)
    {
        int curr_bs = blk_size < (n - s_col) ? blk_size : (n - s_col);
        double *A_scol = A + s_col * ldA;

        // 1. Randomized QR as a preconditioner
        if (use_xr)
        {
            // (1) Randomly select 2 * curr_bs random rows
            memset(flags, 0, sizeof(int) * m);
            for (int i = 0; i < 2 * curr_bs; i++)
            {
                int idx = rand() % m;
                while (flags[idx]) idx = rand() % m;
                flags[idx] = 1;
                P_rowidx_h[i] = idx;
            }
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[0], cublas_stream) );
            CUDA_RT_CHECK( cudaMemcpyAsync(
                P_rowidx_d, P_rowidx_h, sizeof(int) * 2 * curr_bs, 
                cudaMemcpyHostToDevice, cublas_stream
            ) );
            gather_P_rows<<<2 * curr_bs, 32, 0, cublas_stream>>>(
                A_scol, ldA, P, ldP, P_rowidx_d, curr_bs
            );
            CUDA_RT_CHECK( cudaPeekAtLastError() );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[1], cublas_stream) );
            // (2) QR factorization of the selected rows
            CUSOLVER_CHECK( cusolver_stat = cusolverDnDgeqrf(
                cusolver_dn_handle, 2 * curr_bs, curr_bs, P, ldP, tau,
                work_qr, lwork_qr, cusolver_retval
            ) );
            CUDA_RT_CHECK( cudaMemcpyAsync(
                &retval, cusolver_retval, sizeof(int), 
                cudaMemcpyDeviceToHost, cublas_stream
            ) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[2], cublas_stream) );
            // (3) Apply R^-1 to the current panel
            CUBLAS_CHECK( cublasDtrsm(
                cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, curr_bs,
                &d_one, P, ldP, A_scol, ldA
            ) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[3], cublas_stream) );

            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[0]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[1]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[2]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[3]) );
            if (retval != 0)
                fprintf(stderr, "[ERROR] %s, %d: cusolverDnDgeqrf returned %d\n", __FILE__, __LINE__, retval);
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[0], mgs_events[1]) );
            mgs_timers[0] += timer_tmp;
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[1], mgs_events[2]) );
            mgs_timers[1] += timer_tmp;
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[2], mgs_events[3]) );
            mgs_timers[2] += timer_tmp;
        }  // End of "if (use_xr)"

        // 2. CholQR factorization of the current panel
        {
            // (1) Compute the Gram matrix of the current panel
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[0], cublas_stream) );
            CUBLAS_CHECK( cublasDgemm(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                curr_bs, curr_bs, m, &d_one, A_scol, ldA, A_scol, ldA, &d_zero, G, ldG
            ) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[1], cublas_stream) );
            // (2) Cholesky factorization of the Gram matrix
            CUSOLVER_CHECK( cusolverDnDpotrf(
                cusolver_dn_handle, CUBLAS_FILL_MODE_UPPER, curr_bs, G, ldG,
                work_chol, lwork_chol, cusolver_retval
            ) );
            CUDA_RT_CHECK( cudaMemcpyAsync(
                &retval, cusolver_retval, sizeof(int),
                 cudaMemcpyDeviceToHost, cublas_stream
            ) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[2], cublas_stream) );
            // (3) Apply R^-1 to the current panel
            CUBLAS_CHECK( cublasDtrsm(
                cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, curr_bs,
                &d_one, G, ldG, A_scol, ldA
            ) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[3], cublas_stream) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[3], cublas_stream) );

            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[0]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[1]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[2]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[3]) );
            if (retval != 0)
                fprintf(stderr, "[ERROR] %s, %d: cusolverDnDgeqrf returned %d\n", __FILE__, __LINE__, retval);
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[0], mgs_events[1]) );
            mgs_timers[3] += timer_tmp;
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[1], mgs_events[2]) );
            mgs_timers[1] += timer_tmp;
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[2], mgs_events[3]) );
            mgs_timers[2] += timer_tmp;
        }  // End of step 2

        // 3. Subtract current panel from the vectors behind current panel 
        if (s_col + curr_bs < n)
        {
            int n_rem_col = n - (s_col + curr_bs);
            double *A_ecol = A + (s_col + curr_bs) * ldA;
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[0], cublas_stream) );
            CUBLAS_CHECK( cublasDgemm(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                curr_bs, n_rem_col, m, &d_one, A_scol, ldA, A_ecol, ldA, &d_zero, H, ldH
            ) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[1], cublas_stream) );
            CUBLAS_CHECK( cublasDgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                m, n_rem_col, curr_bs, &d_neg_one, A_scol, ldA, H, ldH, &d_one, A_ecol, ldA
            ) );
            CUDA_RT_CHECK( cudaEventRecord(mgs_events[2], cublas_stream) );

            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[0]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[1]) );
            CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[2]) );
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[0], mgs_events[1]) );
            mgs_timers[4] += timer_tmp;
            CUDA_RT_CHECK( cudaEventElapsedTime(&timer_tmp, mgs_events[1], mgs_events[2]) );
            mgs_timers[4] += timer_tmp;
        }  // End of "if (s_col + curr_bs < n)"
    }
}

void test_mgs2(const int m, const int n, const int n_test)
{
    init_cublas_cusolver();

    int blk_size = 512;
    char *mgs_bs_p = getenv("MGS_BS");
    if (mgs_bs_p != NULL) blk_size = atoi(mgs_bs_p);
    if ((blk_size < 32) || (blk_size > 2048)) blk_size = 512;

    int ldA = m, ldR = n;
    double *A = NULL, *A0 = NULL, *A1 = NULL, *R = NULL;
    int *flag = NULL, *P_rowidx_h = NULL, *P_rowidx_d = NULL;
    CUDA_RT_CHECK( cudaMalloc((void **) &A,   sizeof(double) * m * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &A0,  sizeof(double) * m * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &A1,  sizeof(double) * m * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &R,   sizeof(double) * n * n) );
    CUDA_RT_CHECK( cudaMalloc((void **) &P_rowidx_d, sizeof(int) * 2 * blk_size) );
    flag = (int *) malloc(sizeof(int) * m);
    P_rowidx_h = (int *) malloc(sizeof(int) * 2 * blk_size);

    // Calculate the workspace size
    int lwork_qr, lwork_chol, lwork_proj, lwork_max;
    double *workbuf = NULL;
    CUSOLVER_CHECK( cusolverDnDgeqrf_bufferSize(
        cusolver_dn_handle, 2 * blk_size, blk_size, 
        workbuf, 2 * blk_size, &lwork_qr
    ) );
    CUSOLVER_CHECK( cusolverDnDpotrf_bufferSize(
        cusolver_dn_handle, CUBLAS_FILL_MODE_UPPER, blk_size, 
        workbuf, blk_size, &lwork_chol
    ) );
    lwork_proj = blk_size * n;
    lwork_qr += 2 * blk_size * blk_size + blk_size;  // P and tau
    lwork_chol += blk_size * blk_size;  // G
    lwork_max = lwork_qr;
    if (lwork_chol > lwork_max) lwork_max = lwork_chol;
    if (lwork_proj > lwork_max) lwork_max = lwork_proj;
    CUDA_RT_CHECK( cudaMalloc((void **) &workbuf, sizeof(double) * lwork_max ) );
    // Reset lwork_qr and lwork_chol to the original lwork values
    lwork_qr -= 2 * blk_size * blk_size + blk_size;
    lwork_chol -= blk_size * blk_size;
    
    // Initialize random A
    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    CURAND_CHECK( curandGenerateUniformDouble(gen, A0, m * n) );
    CURAND_CHECK( curandDestroyGenerator(gen) );

    // Test MGS2 with randomized XR as panel factorization
    cudaEvent_t *mgs_events = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * 4);
    for (int i = 0; i < 4; i++)
        CUDA_RT_CHECK( cudaEventCreate(&mgs_events[i]) );
    float *mgs_timers = (float *) malloc(sizeof(float) * 6);
    fprintf(stderr, "sample_row, panel_qr/chol, panel_trsm, panel_gemm, proj_gemm  | orth_ms, buildR_ms, orth_gflops, buildR_gflops\n");
    for (int i_test = 0; i_test < n_test; i_test++)
    {
        CUDA_RT_CHECK( cudaMemcpy(A, A0, sizeof(double) * m * n, cudaMemcpyDeviceToDevice) );

        memset(mgs_timers, 0, sizeof(float) * 6);
        MGS(
            m, n, blk_size, A, ldA, 
            P_rowidx_h, P_rowidx_d, flag, 
            workbuf, lwork_qr, lwork_chol, 
            mgs_events, mgs_timers, 1
        );
        MGS(
            m, n, blk_size, A, ldA, 
            P_rowidx_h, P_rowidx_d, flag, 
            workbuf, lwork_qr, lwork_chol, 
            mgs_events, mgs_timers, 0
        );

        float buildR_ms;
        CUDA_RT_CHECK( cudaEventRecord(mgs_events[0], cublas_stream) );
        cublas_stat = cublasDgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, 
            &d_one, A, ldA, A0, ldA, &d_zero, R, ldR
        );
        CUDA_RT_CHECK( cudaEventRecord(mgs_events[1], cublas_stream) );
        CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[0]) );
        CUDA_RT_CHECK( cudaEventSynchronize(mgs_events[1]) );
        CUDA_RT_CHECK( cudaEventElapsedTime(&buildR_ms, mgs_events[0], mgs_events[1]) );

        float orth_mflops = 4.0f * (float) m * (float) n * (float) n / 1000000.0f;
        float buildR_mflops = 2.0f * (float) m * (float) n * (float) n / 1000000.0f;
        float orth_ms = 0.0;
        for (int i = 0; i < 5; i++)
        {
            fprintf(stderr, "%.2f  ", mgs_timers[i]);
            orth_ms += mgs_timers[i];
        }
        fprintf(stderr, "| %.2f, %.2f, %.2f, %.2f\n", orth_ms, buildR_ms, orth_mflops / orth_ms, buildR_mflops / buildR_ms);
    }

    // Accuracy check
    double A0_fnorm, diff_fnorm;
    CUBLAS_CHECK( cublasDgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, 
        &d_one, A, ldA, R, ldR, &d_zero, A1, ldA
    ) );
    CUBLAS_CHECK( cublasDaxpy(cublas_handle, m * n, &d_neg_one, A0, 1, A1, 1) );
    CUBLAS_CHECK( cublasDnrm2(cublas_handle, m * n, A0, 1, &A0_fnorm) );
    CUBLAS_CHECK( cublasDnrm2(cublas_handle, m * n, A1, 1, &diff_fnorm) );
    fprintf(stderr, "MGS2: ||Q * R - A||_{fro} / ||A||_{fro} = %.2e\n", diff_fnorm / A0_fnorm);

    CUDA_RT_CHECK( cudaFree(A) );
    CUDA_RT_CHECK( cudaFree(A0) );
    CUDA_RT_CHECK( cudaFree(R) );
    CUDA_RT_CHECK( cudaFree(P_rowidx_d) );
    CUDA_RT_CHECK( cudaFree(workbuf) );
    for (int i = 0; i < 4; i++)
        CUDA_RT_CHECK( cudaEventDestroy(mgs_events[i]) );
    free(flag);
    free(P_rowidx_h);
}

int main(int argc, char **argv)
{
    int m = 0, n = 0, n_test = 10;
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s m n n_test \n", argv[0]);
        return 255;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    n_test = atoi(argv[3]);
    fprintf(stderr, "m = %d, n = %d, n_test = %d\n", m, n, n_test);

    test_mgs2(m, n, n_test);

    return 0;
}