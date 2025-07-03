import numpy as np
import torch
import math
import argparse
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Test Ozaki Scheme")
    parser.add_argument("--num_split", type=int, default=2, help="Number of splits for the Ozaki Scheme")
    parser.add_argument("--input_dtype", type=str, default="float64", choices=["float64", "float32"], help="Input data type")
    parser.add_argument("--split_dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Data type for splits")
    parser.add_argument("--M", type=int, default=256, help="Number of rows in matrix A")
    parser.add_argument("--N", type=int, default=256, help="Number of columns in matrix B")
    parser.add_argument("--K", type=int, default=256, help="Number of columns in matrix A and rows in matrix B")
    parser.add_argument("--A_bin", type=str, default=None, help="Path to binary file for matrix A (if not using random inputs)")
    parser.add_argument("--B_bin", type=str, default=None, help="Path to binary file for matrix B (if not using random inputs)")
    parser.add_argument("--enable_fast_mode", type=int, default=1, choices=[0, 1], help="Enable fast mode for GEMM")
    parser.add_argument("--split_alg", type=int, default=1, choices=[0, 1], help="FP split algorithm: 0 for paper, 1 for direct")
    return parser.parse_args()

def torch_dtype_from_str(dtype_str):
    if dtype_str == "float64":
        return torch.float64
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported input dtype {dtype_str}")

def check_error(x_ref, x, name=None):
    err = x_ref - x
    abserr = torch.abs(err)
    err_fnorm = torch.norm(err.flatten())
    relerr_fnorm = err_fnorm / torch.norm(x_ref.flatten())
    elem_max_abserr = torch.max(abserr)
    x_ref[x_ref == 0] = 1  # Avoid division by zero
    elem_relerr = abserr / torch.abs(x_ref)
    elem_max_relerr = torch.max(elem_relerr)
    if name is not None:
        print(f"\n{name}:")
        print(f"  Element-wise max abs error : {elem_max_abserr:.9e}")
        print(f"  Element-wise max rel error : {elem_max_relerr:.9e}")
        print(f"  Frobenius norm rel error   : {relerr_fnorm:.9e}")
    return err_fnorm, relerr_fnorm, elem_max_abserr

def os_split_direct(num_split, x, split_dtype=torch.float16, contract_dim=1):
    del contract_dim
    scale = 1.0
    if split_dtype == torch.float16:
        # float16 has 9+1 bits mantissa
        level_scale_exp_coef = 10
        scale = 1024.0
    if split_dtype == torch.bfloat16:
        # bfloat16 has 7+1 bits mantissa
        level_scale_exp_coef = 8
        scale = 256.0
    curr_x = x
    curr_ec = 0
    x_splits = []
    x_ec = []
    for i in range(num_split):
        split_i = curr_x.to(dtype=split_dtype)
        x_splits.append(split_i)
        x_ec.append(curr_ec)
        if i < num_split - 1:
            curr_x = curr_x - split_i.to(dtype=x.dtype)
            curr_x *= scale
            curr_ec -= level_scale_exp_coef
    return x_splits, x_ec

# The function below uses formulas from 10.1007/978-3-030-50743-5_12
def os_split_paper(num_split, x, split_dtype=torch.float16, contract_dim=1):
    curr_x = x
    x_splits = []
    exp_coef = []
    # FP64, FP32, FP16 mantissa bits 53, 24, 10
    cd_bits = math.log2(contract_dim)
    rho = math.ceil(53.0 - min(10.0, (24.0 - cd_bits) * 0.5))
    mu = torch.amax(x)
    for i in range(num_split):
        tau = math.ceil(math.log2(mu))
        sigma = math.exp2(rho + tau)
        x_tmp = (curr_x + sigma) - sigma
        curr_x = curr_x - x_tmp
        split_i = math.exp2(-tau) * x_tmp
        exp_coef.append(tau)
        x_splits.append(split_i.to(dtype=split_dtype))
        if i < num_split - 1:
            mu = torch.amax(curr_x)
    return x_splits, exp_coef

def os_upcast(num_split, x_splits, x_ec, out_dtype=torch.float64):
    output = torch.zeros_like(x_splits[0], dtype=out_dtype)
    for i in range(num_split):
        scale = torch.exp2(torch.tensor(x_ec[i])).to(dtype=out_dtype)
        output += scale * x_splits[i].to(dtype=out_dtype)
    return output

def os_gemm(A_splits, A_ec, B_splits, B_ec, enable_fast_mode=True):
    out_splits = []
    exp_coef = []
    num_splits = min(len(A_splits), len(B_splits))
    for i in range(len(A_splits)):
        for j in range(len(B_splits)):
            if (i + j >= num_splits) and enable_fast_mode:
                continue
            out_ij = torch.matmul(A_splits[i], B_splits[j])
            c_ij = A_ec[i] + B_ec[j]
            out_splits.append(out_ij)
            exp_coef.append(c_ij)
    return out_splits, exp_coef

if __name__ == "__main__":
    np.random.seed(0)
    arg = parse_args()
    num_split = arg.num_split
    enable_fast_mode = arg.enable_fast_mode == 1
    split_alg = arg.split_alg
    os_split = os_split_direct if split_alg == 1 else os_split_paper
    input_dtype = torch_dtype_from_str(arg.input_dtype)
    split_dtype = torch_dtype_from_str(arg.split_dtype)
    M = arg.M
    N = arg.N
    K = arg.K
    if arg.A_bin is not None:
        A = np.fromfile(arg.A_bin, dtype=np.float64).reshape(M, K)
        A = torch.tensor(A, dtype=input_dtype)
    else:
        A = torch.rand((M, K), dtype=input_dtype)
    if arg.B_bin is not None:
        B = np.fromfile(arg.B_bin, dtype=np.float64).reshape(K, N)
        B = torch.tensor(B, dtype=input_dtype)
    else:
        B = torch.rand((K, N), dtype=input_dtype)
    
    if (split_alg == 0) and ((input_dtype != torch.float64) or (split_dtype != torch.float16)):
        raise ValueError("OS paper split algorithm only supports FP64 input and FP16 split dtype.")

    print(f"Testing Ozaki Scheme:")
    print(f"  {input_dtype=}, {split_dtype=}, {num_split=}, {M=}, {N=}, {K=}")
    print(f"  {os_split.__name__=}, {enable_fast_mode=}")
    C = torch.matmul(A, B)

    A_splits, A_ec = os_split(num_split, A, split_dtype=split_dtype, contract_dim=K)
    B_splits, B_ec = os_split(num_split, B, split_dtype=split_dtype, contract_dim=K)
    A_upcast = os_upcast(num_split, A_splits, A_ec, out_dtype=input_dtype)
    B_upcast = os_upcast(num_split, B_splits, B_ec, out_dtype=input_dtype)
    check_error(A, A_upcast, name=f"{num_split} x {arg.split_dtype} upcast A")
    check_error(B, B_upcast, name=f"{num_split} x {arg.split_dtype} upcast B")

    A_splits_fp32 = [a.to(dtype=torch.float32) for a in A_splits]
    B_splits_fp32 = [b.to(dtype=torch.float32) for b in B_splits]
    C_splits, C_ec = os_gemm(A_splits_fp32, A_ec, B_splits_fp32, B_ec, enable_fast_mode=enable_fast_mode)
    C_upcast = os_upcast(len(C_splits), C_splits, C_ec, out_dtype=input_dtype)
    #pdb.set_trace()
    check_error(C, C_upcast, name=f"{num_split} x {arg.split_dtype} FP32 GEMM output C")