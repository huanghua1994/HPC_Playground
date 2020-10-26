__kernel void ocl_daxpy(const int n, const double alpha, __global const double *x,  __global double *y)
{
    //size_t idx = get_global_linear_id(); // This work on NV GPUs but not on Intel CPUs
    int idx = get_global_id(0);
    if (idx < n) y[idx] += alpha * x[idx];
}
