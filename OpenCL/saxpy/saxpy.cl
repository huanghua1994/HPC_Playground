__kernel void my_ocl_saxpy(const int n, const float alpha, __global const float *x,  __global float *y)
{
    size_t idx = get_global_linear_id();
    if (idx < n) y[idx] += alpha * x[idx];
}
