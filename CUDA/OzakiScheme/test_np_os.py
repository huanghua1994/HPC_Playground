import numpy as np
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test Ozaki Scheme with NumPy")
    parser.add_argument("--num_split", type=int, default=2, help="Number of splits for the Ozaki Scheme")
    parser.add_argument("--M", type=int, default=16, help="Number of rows in matrix A")
    parser.add_argument("--N", type=int, default=16, help="Number of columns in matrix B")
    parser.add_argument("--K", type=int, default=64, help="Number of columns in matrix A and rows in matrix B")
    return parser.parse_args()

def os_split(num_split, x, split_dtype=np.float16, contract_dim=1):
    curr_x = x
    x_splits = []
    c_splits = []
    # FP64, FP32, FP16 mantissa bits 52, 23, 10
    cd_bits = math.log2(contract_dim)
    rho = math.ceil(52.0 - min(10.0, (23.0 - cd_bits) * 0.5))
    #mu = np.max(np.abs(x))
    mu = 1.0
    for i in range(num_split):
        tau = math.ceil(math.log2(mu))
        sigma = math.pow(2.0, rho + tau)
        x_tmp = (curr_x + sigma) - sigma
        curr_x = curr_x - x_tmp
        split_i = math.pow(2.0, -tau) * x_tmp
        c_splits.append(tau)
        x_splits.append(split_i.astype(split_dtype))
        if i < num_split - 1:
            #mu = np.max(np.abs(curr_x))
            mu = mu / 1024.0
    return x_splits, c_splits

def os_upcast(num_split, x_splits, c_splits, out_dtype=np.float64):
    output = np.zeros_like(x_splits[0], dtype=out_dtype)
    for i in range(num_split):
        scale = math.pow(2.0, c_splits[i])
        output += scale * x_splits[i].astype(out_dtype)
    return output

def os_gemm(A_splits, A_c, B_splits, B_c, enable_fast_mode=True):
    out_splits = []
    c_splits = []
    num_splits = min(len(A_splits), len(B_splits))
    for i in range(len(A_splits)):
        for j in range(len(B_splits)):
            if (i + j >= num_splits) and enable_fast_mode:
                continue
            out_ij = np.matmul(A_splits[i], B_splits[j])
            c_ij = A_c[i] + B_c[j]
            out_splits.append(out_ij)
            c_splits.append(c_ij)
    return out_splits, c_splits

def check_error(x_ref, x, name=None):
    err = x_ref - x
    abserr = np.abs(err)
    err_fnorm = np.linalg.norm(err, ord='fro')
    relerr_fnorm = err_fnorm / np.linalg.norm(x_ref, ord='fro')
    elem_max_abserr = np.max(abserr)
    elem_max_relerr = np.max(abserr / np.abs(x_ref))
    if name is not None:
        print(f"Array {name}:")
        print(f"  Element-wise max abs error : {elem_max_abserr:.9e}")
        print(f"  Element-wise max rel error : {elem_max_relerr:.9e}")
        print(f"  Frobenius norm rel error   : {relerr_fnorm:.9e}")
    return elem_max_abserr, elem_max_relerr, relerr_fnorm

if __name__ == "__main__":
    np.random.seed(0)
    arg = parse_args()
    num_split = arg.num_split
    M = arg.M
    N = arg.N
    K = arg.K
    print(f"Testing Ozaki Scheme with NumPy: num_split={num_split}, M={M}, N={N}, K={K}")
    A = np.random.rand(M, K)  # float64 by default
    A_splits, A_c = os_split(num_split, A, split_dtype=np.float16, contract_dim=K)
    A_upcast = os_upcast(num_split, A_splits, A_c, out_dtype=np.float64)
    check_error(A, A_upcast, name="Matrix A")

    B = np.random.rand(K, N)
    #A.flatten(order='F').tofile(f"A_{M}x{K}.bin")
    #B.flatten(order='F').tofile(f"B_{K}x{N}.bin")
    C = np.matmul(A, B)
    B_splits, B_c = os_split(num_split, B, split_dtype=np.float16, contract_dim=N)
    C_splits, C_c = os_gemm(A_splits, A_c, B_splits, B_c)
    C_upcast = os_upcast(len(C_splits), C_splits, C_c, out_dtype=np.float64)
    check_error(C, C_upcast, name="FP16 split FP16 GEMM output C")
    A_splits_f16_f32 = [a.astype(np.float32) for a in A_splits]
    B_splits_f16_f32 = [b.astype(np.float32) for b in B_splits]
    C_splits_f16_f32, C_c2 = os_gemm(A_splits_f16_f32, A_c, B_splits_f16_f32, B_c)
    C_upcast_f16_f32 = os_upcast(len(C_splits_f16_f32), C_splits_f16_f32, C_c2, out_dtype=np.float64)
    check_error(C, C_upcast_f16_f32, name="FP16 split FP32 GEMM output C")

