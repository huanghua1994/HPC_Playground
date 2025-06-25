import numpy as np
from numpy_ozaki_scheme import np_os_split, np_os_upcast, np_os_gemm, np_check_error

if __name__ == "__main__":
    np.random.seed(0)
    M = 16
    N = 16
    K = 64
    num_split = 2
    A = np.random.rand(M, K)  # float64 by default
    A[0, 0] = -2.0/3.0
    A_splits = np_os_split(num_split, A, split_dtype=np.float16)
    A_upcast = np_os_upcast(num_split, A_splits, out_dtype=np.float64)
    np_check_error(A, A_upcast, name="Matrix A")

    B = np.random.rand(K, N)
    C = np.matmul(A, B)
    B_splits = np_os_split(num_split, B, split_dtype=np.float16)
    C_splits = np_os_gemm(num_split, A_splits, B_splits)
    C_upcast = np_os_upcast(num_split, C_splits, out_dtype=np.float64)
    np_check_error(C, C_upcast, name="FP16 split FP16 GEMM output C")
    A_splits_f16_f32 = [a.astype(np.float32) for a in A_splits]
    B_splits_f16_f32 = [b.astype(np.float32) for b in B_splits]
    C_splits_f16_f32 = np_os_gemm(num_split, A_splits_f16_f32, B_splits_f16_f32)
    C_upcast_f16_f32 = np_os_upcast(num_split, C_splits_f16_f32, out_dtype=np.float64)
    np_check_error(C, C_upcast_f16_f32, name="FP16 split FP32 GEMM output C")

