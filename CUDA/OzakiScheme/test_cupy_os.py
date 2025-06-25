import cupy
from cupy_ozaki_scheme import cupy_os_split, cupy_os_upcast, cupy_os_gemm, cupy_check_error

if __name__ == "__main__":
    cupy.random.seed(0)
    M = 16
    N = 16
    K = 64
    num_split = 2
    A = cupy.random.rand(M, K)  # float64 by default
    A[0, 0] = -2.0/3.0
    A_splits = cupy_os_split(num_split, A, split_dtype=cupy.float16)
    A_upcast = cupy_os_upcast(num_split, A_splits, out_dtype=cupy.float64)
    cupy_check_error(A, A_upcast, name="Matrix A")

    B = cupy.random.rand(K, N)
    C = cupy.matmul(A, B)
    B_splits = cupy_os_split(num_split, B, split_dtype=cupy.float16)
    C_splits = cupy_os_gemm(num_split, A_splits, B_splits)
    C_upcast = cupy_os_upcast(num_split, C_splits, out_dtype=cupy.float64)
    cupy_check_error(C, C_upcast, name="FP16 split FP16 GEMM output C")
    A_splits_f16_f32 = [a.astype(cupy.float32) for a in A_splits]
    B_splits_f16_f32 = [b.astype(cupy.float32) for b in B_splits]
    C_splits_f16_f32 = cupy_os_gemm(num_split, A_splits_f16_f32, B_splits_f16_f32)
    C_upcast_f16_f32 = cupy_os_upcast(num_split, C_splits_f16_f32, out_dtype=cupy.float64)
    cupy_check_error(C, C_upcast_f16_f32, name="FP16 split FP32 GEMM output C")

