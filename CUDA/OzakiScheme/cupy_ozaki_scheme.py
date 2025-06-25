import cupy

def cupy_os_split(num_split, x, split_dtype=cupy.float16):
    scale = 1024.0
    curr_x = x
    x_splits = []
    for i in range(num_split):
        split_i = curr_x.astype(split_dtype)
        x_splits.append(split_i)
        if i < num_split - 1:
            curr_x = (curr_x - split_i.astype(x.dtype)) * scale
    return x_splits

def cupy_os_upcast(num_split, x_splits, out_dtype=cupy.float64):
    scale = 1024.0
    curr_scale_inv = 1.0
    output = cupy.zeros_like(x_splits[0], dtype=out_dtype)
    for i in range(num_split):
        output += x_splits[i].astype(out_dtype) * curr_scale_inv
        if i < num_split - 1:
            curr_scale_inv /= scale
    return output

def cupy_os_gemm(num_split, A_splits, B_splits):
    if num_split == 1:
        return [cupy.matmul(A_splits[0], B_splits[0])]
    out_shape = (A_splits[0].shape[0], B_splits[0].shape[1])
    out_splits = [cupy.zeros(out_shape, dtype=A_splits[0].dtype) for _ in range(num_split)]
    for i in range(num_split):
        for j in range(num_split - i):
            k = i + j
            out_splits[k] += cupy.matmul(A_splits[i], B_splits[j])
    return out_splits

def cupy_check_error(x_ref, x, name=None):
    err = x_ref - x
    abserr = cupy.abs(err)
    err_fnorm = cupy.linalg.norm(err, ord='fro')
    relerr_fnorm = err_fnorm / cupy.linalg.norm(x_ref, ord='fro')
    elem_max_abserr = cupy.max(abserr)
    elem_max_relerr = cupy.max(abserr / cupy.abs(x_ref))
    if name is not None:
        print(f"Array {name}:")
        print(f"  Element-wise max abs error : {elem_max_abserr:.9e}")
        print(f"  Element-wise max rel error : {elem_max_relerr:.9e}")
        print(f"  Frobenius norm rel error   : {relerr_fnorm:.9e}")
    return elem_max_abserr, elem_max_relerr, relerr_fnorm
