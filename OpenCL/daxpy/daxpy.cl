__kernel void ocl_daxpy(const int n, const double alpha, __global const double *x,  __global double *y)
{
    size_t idx = get_global_linear_id();
    if (idx < n) y[idx] += alpha * x[idx];
}
