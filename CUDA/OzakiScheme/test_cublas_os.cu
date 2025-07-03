#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

/*
nvcc test_cublas_os.cu -O2 -gencode arch=compute_70,code=sm_70 -gencode \
    arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -o test_cublas_os.exe -lcublas -lcublasLt -lm
*/

#define CUDA_CHECK(statement)                                                       \
    do                                                                              \
    {                                                                               \
        cudaError_t result = (statement);                                           \
        if (cudaSuccess != result)                                                  \
        {                                                                           \
            fprintf(stderr, "[%s:%d] CUDA failed, ", __FUNCTION__, __LINE__);       \
            fprintf(stderr, "reason: %s", cudaGetErrorString(result));              \
        }                                                                           \
        assert(cudaSuccess == result);                                              \
    } while (0)


#define CUBLAS_CHECK(statement)                                                     \
    do                                                                              \
    {                                                                               \
        cublasStatus_t result = (statement);                                        \
        if (CUBLAS_STATUS_SUCCESS != result)                                        \
        {                                                                           \
            fprintf(stderr, "[%s:%d] cuBLAS failed, ", __FUNCTION__, __LINE__);     \
            fprintf(stderr, "ret = %d\n", result);                                  \
        }                                                                           \
        assert(CUBLAS_STATUS_SUCCESS == result);                                    \
    } while (0)

#define ERROR_CHECK(statement, failed_reason)                                       \
    do                                                                              \
    {                                                                               \
        int result = (statement);                                                   \
        if (!result)                                                                \
        {                                                                           \
            fprintf(stderr, "[%s:%d] Check failed, ", __FUNCTION__, __LINE__);      \
            fprintf(stderr, "reason: %s\n", failed_reason);                         \
        }                                                                           \
        assert(result);                                                             \
    } while (0)

template<int num_split>
__device__ void os_fp_split_direct(
    double curr_x, const int rho_int, __half *output_0,
    __half *output_1, __half *output_2, __half *output_3
)
{
    // After each split, we scale up the residual by 2^10 since
    // FP16 e5m9 has 10 bits of effective precision.

    __half out0 = static_cast<__half>(curr_x);
    *output_0 = out0;
    if constexpr(num_split == 1) return;

    curr_x -= static_cast<double>(out0);
    curr_x = scalbn(curr_x, 10);
    __half out1 = static_cast<__half>(curr_x);
    *output_1 = out1;
    if constexpr(num_split == 2) return;

    curr_x -= static_cast<double>(out1);
    curr_x = scalbn(curr_x, 10);
    __half out2 = static_cast<__half>(curr_x);
    *output_2 = out2;
    if constexpr(num_split == 3) return;

    curr_x -= static_cast<double>(out2);
    curr_x = scalbn(curr_x, 10);
    __half out3 = static_cast<__half>(curr_x);
    *output_3 = out3;
}

// The function below uses formulas from 10.1007/978-3-030-50743-5_12
template<int num_split>
__device__ void os_fp_split_paper(
    double curr_x, const int rho_int, __half *output_0,
    __half *output_1, __half *output_2, __half *output_3
)
{
    int tau_int = 0;
    double sigma, x_tmp, out_value;

    // After each split, we scale up the residual by 2^10 since 
    // FP16 has 10 bits of effective precision.
    
    sigma = scalbn(1.0, rho_int + tau_int);
    x_tmp = (curr_x + sigma) - sigma;
    out_value = scalbn(x_tmp, -tau_int);
    *output_0 = static_cast<__half>(out_value);
    if constexpr(num_split == 1) return;

    curr_x -= x_tmp;
    tau_int -= 10;
    sigma = scalbn(sigma, -10);
    x_tmp = (curr_x + sigma) - sigma;
    out_value = scalbn(x_tmp, -tau_int);
    *output_1 = static_cast<__half>(out_value);
    if constexpr(num_split == 2) return;

    curr_x -= x_tmp;
    tau_int -= 10;
    sigma = scalbn(sigma, -10);
    x_tmp = (curr_x + sigma) - sigma;
    out_value = scalbn(x_tmp, -tau_int);
    *output_2 = static_cast<__half>(out_value);
    if constexpr(num_split == 3) return;

    curr_x -= x_tmp;
    tau_int -= 10;
    sigma = scalbn(sigma, -10);
    x_tmp = (curr_x + sigma) - sigma;
    out_value = scalbn(x_tmp, -tau_int);
    *output_3 = static_cast<__half>(out_value);
}

template<int num_split, typename InType, typename SplitType>
__global__ void split_array_kernel(
    const int arr_size, const InType* __restrict__ input,
    SplitType* __restrict__ output_0, SplitType* __restrict__ output_1,
    SplitType* __restrict__ output_2, SplitType* __restrict__ output_3,
    InType contract_dim = 1.0
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= arr_size) return;
    InType cd_bits = log2(contract_dim);
    InType rho = ceil(53.0 - min(10.0, (24.0 - cd_bits) * 0.5));
    int rho_int = static_cast<int>(rho);
    os_fp_split_direct<num_split>(
        input[idx], rho_int, 
        output_0 + idx, output_1 + idx, 
        output_2 + idx, output_3 + idx
    );
}


template<int num_split, typename SplitType, typename OutType>
__global__ void sum_split_array_kernel(
    const int arr_size, OutType* __restrict__ output,
    SplitType* __restrict__ split_0, SplitType* __restrict__ split_1,
    SplitType* __restrict__ split_2, SplitType* __restrict__ split_3,
    OutType level_scale_inv = 1.0
)
{
    if (level_scale_inv == 1.0)
    {
        // FP16: e5m9, 10 bits of effective precision
        if (std::is_same_v<SplitType, __half>) level_scale_inv = 1.0 / 1024.0;
        // FP32: e8m23, 24 bits of effective precision
        if (std::is_same_v<SplitType, float>) level_scale_inv = 1.0 / 16777216.0;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= arr_size) return;
    OutType out_value = 0;
    OutType curr_scale_inv = 1.0;
    out_value += static_cast<OutType>(split_0[idx]);
    if constexpr(num_split >= 2)
    {
        curr_scale_inv *= level_scale_inv;
        out_value += curr_scale_inv * static_cast<OutType>(split_1[idx]);
    }
    if constexpr(num_split >= 3)
    {
        curr_scale_inv *= level_scale_inv;
        out_value += curr_scale_inv * static_cast<OutType>(split_2[idx]);
    }
    if constexpr(num_split >= 4)
    {
        curr_scale_inv *= level_scale_inv;
        out_value += curr_scale_inv * static_cast<OutType>(split_3[idx]);
    }
    output[idx] = out_value;
}

template<typename T>
std::vector<T> dev_array_to_host(const void *dptr, size_t n_elem)
{
    std::vector<T> data(n_elem);
    if (n_elem == 0) return data;

    T *hptr = data.data();
    CUDA_CHECK( cudaMemcpy(hptr, dptr, n_elem * sizeof(T), cudaMemcpyDeviceToHost) );
    return data;
}

template<typename T>
std::vector<T> read_binary_file(const char *fname, int n_elem)
{
    std::vector<T> data;
    FILE *file = fopen(fname, "rb");
    if (!file)
    {
        fprintf(stderr, "Failed to open file: %s\n", fname);
        return data;
    }
    data.resize(n_elem);
    int read_size = fread(data.data(), sizeof(T), n_elem, file);
    if (read_size != n_elem)
    {
        fprintf(stderr, "Failed to read %d elements from file: %s\n", n_elem, fname);
        data.clear();
        return data;
    }
    fclose(file);
    return data;
}

template<typename T>
void write_binary_file(const char *fname, int n_elem, T *data)
{
    FILE *file = fopen(fname, "wb");
    if (!file)
    {
        fprintf(stderr, "Failed to open file for writing: %s\n", fname);
        return;
    }
    int written_size = fwrite(data, sizeof(T), n_elem, file);
    if (written_size != n_elem)
        fprintf(stderr, "Failed to write %d elements to file: %s\n", n_elem, fname);
    fclose(file);
}

template<typename T>
void print_colmajor_matrix(const char *name, const T *mat, const int ldm, int nrow, int ncol)
{
    printf("%s:\n", name);
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < ncol; j++)
            printf("% 10.8f ", mat[i + j * ldm]);
        printf("\n");
    }
}

void cublaslt_gemm(
    cudaStream_t stream, cublasLtHandle_t lt_handle,
    const int m, const int n, const int k,
    cublasOperation_t transA, cublasOperation_t transB,
    cublasComputeType_t compute_type, cudaDataType_t scale_type,
    cudaDataType_t AB_type, cudaDataType_t C_type,
    void *A, const int ldA, void *B, const int ldB, void *C, const int ldC,
    const int init_output, void *workspace, const size_t workspace_bytes
)
{
    int return_result = 0;
    cublasLtMatmulDesc_t matmul_desc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heur_result = {};

    double f64_one = 1, f64_zero = 0;
    float  f32_one = 1, f32_zero = 0;

    CUBLAS_CHECK( cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type) );
    CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) );
    CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) );

    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Adesc, AB_type, transA == CUBLAS_OP_N ? m : k, transA == CUBLAS_OP_N ? k : m, ldA) );
    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Bdesc, AB_type, transB == CUBLAS_OP_N ? k : n, transB == CUBLAS_OP_N ? n : k, ldB) );
    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Cdesc, C_type, m, n, ldC));

    CUBLAS_CHECK( cublasLtMatmulPreferenceCreate(&preference) );
    CUBLAS_CHECK( cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)
    ) );
    CUBLAS_CHECK( cublasLtMatmulAlgoGetHeuristic(
        lt_handle, matmul_desc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heur_result, &return_result
    ) );
    if (return_result == 0) CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);

    void *alpha = nullptr, *beta = nullptr;
    if (AB_type == CUDA_R_64F)
    {
        alpha = reinterpret_cast<void *>(&f64_one);
        beta = init_output ? reinterpret_cast<void *>(&f64_zero) : alpha;
    } else {
        alpha = reinterpret_cast<void *>(&f32_one);
        beta = init_output ? reinterpret_cast<void *>(&f32_zero) : alpha;
    }

    CUBLAS_CHECK( cublasLtMatmul(
        lt_handle, matmul_desc, alpha, A, Adesc, B, Bdesc,
        beta, C, Cdesc, C, Cdesc, &heur_result.algo,
        workspace, workspace_bytes, stream
    ) );
}

void calc_error(
    const int n_elem, const double *x_ref, const double *x, 
    double *elem_max_abs_err, double *elem_max_rel_err, double *mat_relerr
)
{
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double mat_rel_err = 0.0;
    double mat_fnorm = 0.0;

    for (int i = 0; i < n_elem; i++)
    {
        double abs_err = fabs(x[i] - x_ref[i]);
        double rel_err = (fabs(x_ref[i]) > 1e-12) ? (abs_err / fabs(x_ref[i])) : 0.0;
        max_abs_err = fmax(max_abs_err, abs_err);
        max_rel_err = fmax(max_rel_err, rel_err);
        mat_fnorm += x_ref[i] * x_ref[i];
        mat_rel_err += abs_err * abs_err;
    }
    mat_fnorm = sqrt(mat_fnorm);
    mat_rel_err = sqrt(mat_rel_err) / mat_fnorm;

    *elem_max_abs_err = max_abs_err;
    *elem_max_rel_err = max_rel_err;
    *mat_relerr = mat_rel_err;
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        fprintf(stderr, "Usage: %s <num_split> <m> <n> <k> <A-binary> <B-binary>\n", argv[0]);
        return -1;
    }
    int num_split = atoi(argv[1]);
    if (num_split < 1 || num_split > 4) num_split = 1;
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int k = atoi(argv[4]);
    printf("num_split = %d, m = %d, n = %d, k = %d\n", num_split, m, n, k);
    int ldA = m, ldB = k, ldC = m;
    auto transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
    double max_abs_err = 0.0, max_rel_err = 0.0, mat_relerr = 0.0;

    cudaStream_t stream;
    cublasLtHandle_t lt_handle;
    CUDA_CHECK( cudaStreamCreate(&stream) );
    CUBLAS_CHECK( cublasLtCreate(&lt_handle) );

    double *A_fp64 = nullptr, *B_fp64 = nullptr, *C_fp64 = nullptr, *C_upcast_fp64 = nullptr;
    __half *A_splits_fp16 = nullptr, *B_splits_fp16 = nullptr;
    float *C_splits_fp32 = nullptr;
    void *workspace = nullptr;
    size_t workspace_bytes = 32 * 1024 * 1024;
    int A_size = m * k, B_size = k * n, C_size = m * n;
    CUDA_CHECK( cudaMalloc((void **) &A_fp64, A_size * sizeof(double)) );
    CUDA_CHECK( cudaMalloc((void **) &B_fp64, B_size * sizeof(double)) );
    CUDA_CHECK( cudaMalloc((void **) &C_fp64, C_size * sizeof(double)) );
    CUDA_CHECK( cudaMalloc((void **) &C_upcast_fp64, C_size * sizeof(double)) );
    CUDA_CHECK( cudaMalloc((void **) &A_splits_fp16, A_size * sizeof(__half) * num_split) );
    CUDA_CHECK( cudaMalloc((void **) &B_splits_fp16, B_size * sizeof(__half) * num_split) );
    CUDA_CHECK( cudaMalloc((void **) &C_splits_fp32, C_size * sizeof(float) * num_split) );
    CUDA_CHECK( cudaMalloc(&workspace, workspace_bytes) );
    printf("CUDA memory allocated\n");

    auto input_A = read_binary_file<double>(argv[5], A_size);
    auto input_B = read_binary_file<double>(argv[6], B_size);
    double *A_fp64_h = input_A.data();
    double *B_fp64_h = input_B.data();
    CUDA_CHECK( cudaMemcpy(A_fp64, A_fp64_h, A_size * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(B_fp64, B_fp64_h, B_size * sizeof(double), cudaMemcpyHostToDevice) );
    printf("Input matrices A and B loaded and copied to GPU\n");

    dim3 block(256);
    dim3 A_mat_grid((A_size + block.x - 1) / block.x);
    dim3 B_mat_grid((B_size + block.x - 1) / block.x);
    dim3 C_mat_grid((C_size + block.x - 1) / block.x);
    double contract_dim = static_cast<double>(k);
    #define DISPATCH_SPLIT_ARRAY_KERNEL(num_split) \
    do {  \
        split_array_kernel<num_split, double, __half><<<A_mat_grid, block, 0, stream>>>(  \
            A_size, A_fp64, A_splits_fp16, A_splits_fp16 + A_size,  \
            A_splits_fp16 + 2 * A_size, A_splits_fp16 + 3 * A_size,  \
            contract_dim  \
        );  \
        split_array_kernel<num_split, double, __half><<<B_mat_grid, block, 0, stream>>>(  \
            B_size, B_fp64, B_splits_fp16, B_splits_fp16 + B_size,  \
            B_splits_fp16 + 2 * B_size, B_splits_fp16 + 3 * B_size,  \
            contract_dim  \
        );  \
    } while (0)
    if (num_split == 1) DISPATCH_SPLIT_ARRAY_KERNEL(1);
    else if (num_split == 2) DISPATCH_SPLIT_ARRAY_KERNEL(2);
    else if (num_split == 3) DISPATCH_SPLIT_ARRAY_KERNEL(3);
    else if (num_split == 4) DISPATCH_SPLIT_ARRAY_KERNEL(4);
    else
        fprintf(stderr, "Invalid num_split: %d\n", num_split);
    CUDA_CHECK( cudaGetLastError() );
    printf("Input matrices A and B split into %d parts\n", num_split);
    #undef DISPATCH_SPLIT_ARRAY_KERNEL

    double *A_upcast_fp64 = nullptr;
    CUDA_CHECK( cudaMalloc((void **) &A_upcast_fp64, A_size * sizeof(double)) );
    #define DISPATCH_SUM_SPLIT_ARRAY_KERNEL(num_split) \
    do {  \
        sum_split_array_kernel<num_split, __half, double><<<A_mat_grid, block, 0, stream>>>(  \
            A_size, A_upcast_fp64, A_splits_fp16, A_splits_fp16 + A_size,  \
            A_splits_fp16 + 2 * A_size, A_splits_fp16 + 3 * A_size, 1.0/1024.0  \
        );  \
    } while (0)
    if (num_split == 1) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(1);
    else if (num_split == 2) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(2);
    else if (num_split == 3) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(3);
    else if (num_split == 4) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(4);
    else
        fprintf(stderr, "Invalid num_split: %d\n", num_split);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaStreamSynchronize(stream) );
    #undef DISPATCH_SUM_SPLIT_ARRAY_KERNEL
    auto A_upcast_fp64_h = dev_array_to_host<double>(A_upcast_fp64, A_size);
    calc_error(
        A_size, A_fp64_h, A_upcast_fp64_h.data(),
        &max_abs_err, &max_rel_err, &mat_relerr
    );
    printf("A split summed up:\n");
    printf("  max_abs_err = %e\n  max_rel_err = %e\n  mat_relerr  = %e\n",
           max_abs_err, max_rel_err, mat_relerr);

    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t AB_type = CUDA_R_16F, C_type = CUDA_R_32F, scale_type = CUDA_R_32F;
    for (int i = 0; i < num_split; i++)
    {
        __half *A_split_i = A_splits_fp16 + i * A_size;
        int init_output = (i == 0);
        for (int j = 0; j < num_split - i; j++)
        {
            int k_ = i + j;
            __half *B_split_j = B_splits_fp16 + j * B_size;
            float *C_split_k = C_splits_fp32 + k_ * C_size;
            cublaslt_gemm(
                stream, lt_handle, m, n, k, transA, transB,
                compute_type, scale_type, AB_type, C_type,
                A_split_i, ldA, B_split_j, ldB, C_split_k, ldC,
                init_output, workspace, workspace_bytes
            );
            CUDA_CHECK( cudaGetLastError() );
        }
    }
    printf("cublasLt GEMM with split arrays completed\n");

    #define DISPATCH_SUM_SPLIT_ARRAY_KERNEL(num_split) \
    do {  \
        sum_split_array_kernel<num_split, float, double><<<C_mat_grid, block, 0, stream>>>(  \
            C_size, C_upcast_fp64, C_splits_fp32, C_splits_fp32 + C_size,  \
            C_splits_fp32 + 2 * C_size, C_splits_fp32 + 3 * C_size, 1.0/1024.0  \
        );  \
    } while (0)
    if (num_split == 1) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(1);
    else if (num_split == 2) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(2);
    else if (num_split == 3) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(3);
    else if (num_split == 4) DISPATCH_SUM_SPLIT_ARRAY_KERNEL(4);
    else
        fprintf(stderr, "Invalid num_split: %d\n", num_split);
    CUDA_CHECK( cudaGetLastError() );
    printf("Split arrays summed up to C_upcast_fp64\n");
    #undef DISPATCH_SUM_SPLIT_ARRAY_KERNEL

    compute_type = CUBLAS_COMPUTE_64F;
    AB_type = CUDA_R_64F;
    C_type = CUDA_R_64F;
    scale_type = CUDA_R_64F;
    cublaslt_gemm(
        stream, lt_handle, m, n, k, transA, transB,
        compute_type, scale_type, AB_type, C_type,
        A_fp64, ldA, B_fp64, ldB, C_fp64, ldC,
        1, workspace, workspace_bytes
    );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaStreamSynchronize(stream) );
    printf("cublasLt GEMM with full precision matrices A and B completed\n");

    auto C_fp64_h = dev_array_to_host<double>(C_fp64, C_size);
    auto C_upcast_fp64_h = dev_array_to_host<double>(C_upcast_fp64, C_size);
    calc_error(
        C_size, C_fp64_h.data(), C_upcast_fp64_h.data(),
        &max_abs_err, &max_rel_err, &mat_relerr
    );
    printf("FP16 split FP32 GEMM output C:\n");
    printf("  max_abs_err = %e\n  max_rel_err = %e\n  mat_relerr  = %e\n",
           max_abs_err, max_rel_err, mat_relerr);

    CUDA_CHECK( cudaFree(A_fp64) );
    CUDA_CHECK( cudaFree(B_fp64) );
    CUDA_CHECK( cudaFree(C_fp64) );
    CUDA_CHECK( cudaFree(C_upcast_fp64) );
    CUDA_CHECK( cudaFree(A_splits_fp16) );
    CUDA_CHECK( cudaFree(B_splits_fp16) );
    CUDA_CHECK( cudaFree(C_splits_fp32) );
    CUDA_CHECK( cudaFree(workspace) );
    return 0;
}