#pragma once

#include <iostream>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include "householder_qr.cuh"

template <typename T>
__global__ void zero_lower_triangle_kernel(T *A, const int ldA, int m, int n) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n && row > col)
    {
        size_t offset = (size_t) row + (size_t) col * (size_t) ldA;
        A[offset] = 0;
    }
}

template <typename T>
void zero_lower_triangle(T *A, const int ldA, int m, int n)
{
    dim3 block(32, 32);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    zero_lower_triangle_kernel<<<grid, block>>>(A, ldA, m, n);
}

template <typename T>
void print_dev_matrix(const char *name, T *A, const int ldA, int m, int n)
{
    T *A_h = (T *) malloc(sizeof(T) * m * n);
    std::cout << "Matrix " << name << ":" << std::endl;
    CUDA_CHECK( cudaMemcpy(A_h, A, sizeof(T) * m * n, cudaMemcpyDeviceToHost) );
    if (std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) printf("% .4e ", A_h[i + j * ldA]);
            std::cout << std::endl;
        }
    } else {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
                std::cout << A_h[i + j * ldA] << " ";
            std::cout << std::endl;
        }
    }
    free(A_h);
}

template <typename T>
void rand_mat_cond_num(const int m, const int n, T *A, const int ldA, T cond_num)
{
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_dn_handle;
    cusolverDnParams_t cusolver_dn_params;
    CUDA_CHECK( cudaStreamCreate(&stream) );
    CUBLAS_CHECK( cublasCreate(&cublas_handle) );
    CUSOLVER_CHECK( cusolverDnCreate(&cusolver_dn_handle) );
    CUSOLVER_CHECK( cusolverDnCreateParams(&cusolver_dn_params) );

    constexpr bool T_is_double = std::is_same_v<T, double>;
    constexpr bool T_is_float  = std::is_same_v<T, float>;

    // Allocate work arrays
    const int ldU = m, ldV = n, ldW = m;
    T *U0 = nullptr, *U = nullptr, *S = nullptr, *V0 = nullptr, *V = nullptr, *tau = nullptr, *W = nullptr;
    CUDA_CHECK( cudaMalloc((void **) &U0,  sizeof(V) * ldU * n) );
    CUDA_CHECK( cudaMalloc((void **) &U,   sizeof(V) * ldU * n) );
    CUDA_CHECK( cudaMalloc((void **) &S,   sizeof(V) * n) );
    CUDA_CHECK( cudaMalloc((void **) &V0,  sizeof(V) * ldV * n) );
    CUDA_CHECK( cudaMalloc((void **) &V,   sizeof(V) * ldV * n) );
    CUDA_CHECK( cudaMalloc((void **) &tau, sizeof(V) * n) );
    CUDA_CHECK( cudaMalloc((void **) &W,   sizeof(T) * ldW * n) );

    // Generate random orthogonal matrices U and V
    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, 19241112) );
    if constexpr(T_is_double)
    {
        CURAND_CHECK( curandGenerateUniformDouble(gen, U0, ldU * n) );
        CURAND_CHECK( curandGenerateUniformDouble(gen, V0, ldV * n) );
    }
    if constexpr(T_is_float)
    {
        CURAND_CHECK( curandGenerateUniform(gen, U0, ldU * n) );
        CURAND_CHECK( curandGenerateUniform(gen, V0, ldV * n) );
    }
    CURAND_CHECK( curandDestroyGenerator(gen) );
    householder_qr_workbuf_s *U_workbuf = (householder_qr_workbuf_s *) malloc(sizeof(householder_qr_workbuf_s));
    householder_qr_workbuf_s *V_workbuf = (householder_qr_workbuf_s *) malloc(sizeof(householder_qr_workbuf_s));
    householder_qr_workbuf_alloc<T>(cusolver_dn_handle, cusolver_dn_params, m, n, U_workbuf);
    householder_qr_workbuf_alloc<T>(cusolver_dn_handle, cusolver_dn_params, n, n, V_workbuf);
    int explicit_Q = 1;
    householder_qr<T>(
        m, n, U0, ldU, tau, explicit_Q, U, ldU,
        U_workbuf, stream, cusolver_dn_handle, cusolver_dn_params
    );
    householder_qr<T>(
        n, n, V0, ldV, tau, explicit_Q, V, ldV,
        V_workbuf, stream, cusolver_dn_handle, cusolver_dn_params
    );

    // Update S to have condition number cond_num
    if (T_is_double && cond_num > 1e15) cond_num = 1e15;
    if (T_is_float  && cond_num > 1e6)  cond_num = 1e6;
    T *S_h = (T *) malloc(sizeof(T) * n);
    CUDA_CHECK( cudaMemcpy(S_h, S, sizeof(T) * n, cudaMemcpyDeviceToHost) );
    T S1 = 1.0 / cond_num;
    T Sn = 1.0;
    T d = (S1 - Sn) / (T) (n - 1);
    for (int i = 0; i < n; i++) S_h[i] = S1 + (T) i * d;
    CUDA_CHECK( cudaMemcpy(S, S_h, sizeof(T) * n, cudaMemcpyHostToDevice) );

    // A = U * S * V^T = W * V^T, where W = U * S
    if constexpr(T_is_double)
    {
        double d_one = 1.0, d_zero = 0.0;
        CUBLAS_CHECK( cublasDdgmm(cublas_handle, CUBLAS_SIDE_RIGHT, m, n, U, ldU, S, 1, W, ldW));
        CUBLAS_CHECK( cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            m, n, n, &d_one, W, ldW, V, ldV, &d_zero, A, ldA
        ) );
    }
    if constexpr(T_is_float)
    {
        float s_one = 1.0, s_zero = 0.0;
        CUBLAS_CHECK( cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT, m, n, U, ldU, S, 1, W, ldW));
        CUBLAS_CHECK( cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            m, n, n, &s_one, W, ldW, V, ldV, &s_zero, A, ldA
        ) );
    }

    // Clean up
    CUDA_CHECK( cudaFree(U0) );
    CUDA_CHECK( cudaFree(U) );
    CUDA_CHECK( cudaFree(S) );
    CUDA_CHECK( cudaFree(V0) );
    CUDA_CHECK( cudaFree(V) );
    CUDA_CHECK( cudaFree(tau) );
    CUDA_CHECK( cudaFree(W) );
    CUDA_CHECK( cudaStreamDestroy(stream) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUSOLVER_CHECK( cusolverDnDestroy(cusolver_dn_handle) );
    CUSOLVER_CHECK( cusolverDnDestroyParams(cusolver_dn_params) );
}
