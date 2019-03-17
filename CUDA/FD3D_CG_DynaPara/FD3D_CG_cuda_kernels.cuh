// Note: If you change the RADIUS, you should also change the unrolling below
#define RADIUS 6

#define X_BLK_SIZE 32
#define Y_BLK_SIZE 16
#define X_BLK_2R   (X_BLK_SIZE + 2 * (RADIUS))
#define Y_BLK_2R   (Y_BLK_SIZE + 2 * (RADIUS))

__constant__ double cu_coef[RADIUS + 1];

__global__ void stencil_3axis_cuda_kernel(
    const int nx, const int ny, const int nz, 
    const double *x0, double *x1
)
{
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int local_y  = threadIdx.y;
    const int stride_z = nx * ny;
    const int stride_y_ex = nx + 2 * RADIUS;
    const int stride_z_ex = stride_y_ex * (ny + 2 * RADIUS);
    const int tile_x = local_x + RADIUS;
    const int tile_y = local_y + RADIUS;
    const double coef0 = cu_coef[0] * 3.0;
    
    int thread_x0_idx, thread_x0_ridx, thread_x1_idx;
    thread_x0_idx  = RADIUS * stride_z_ex;
    thread_x0_idx += (global_y + RADIUS) * stride_y_ex;
    thread_x0_idx += (global_x + RADIUS);
    thread_x1_idx  = global_y * nx + global_x;
    thread_x0_ridx = thread_x0_idx - RADIUS * stride_z_ex;
    
    bool valid_x1 = true;
    if ((global_x >= nx) || (global_y >= ny)) valid_x1 = false;
    
    double z_axis_buff[2 * RADIUS + 1], current_x0_z;
    __shared__ double xy_plane[Y_BLK_2R][X_BLK_2R];
    
    // Prefetch z-axis front and behind data
    for (int iz = -RADIUS; iz < RADIUS; iz++)
    {
        // +1 here because we will advance the z index first
        z_axis_buff[iz + RADIUS + 1] = x0[thread_x0_ridx];
        thread_x0_ridx += stride_z_ex;
    }
    
    // Step through the xy-planes
    for (int iz = 0; iz < nz; iz++)
    {
        // 1. Advance the z-axis thread buffer
        #pragma unroll 12
        for (int i = 0; i < 2 * RADIUS; i++)
            z_axis_buff[i] = z_axis_buff[i + 1];
        
        if (valid_x1) z_axis_buff[2 * RADIUS] = x0[thread_x0_ridx];
        thread_x0_ridx += stride_z_ex;
        
        __syncthreads();
        
        // 2. Load the x & y halo for current z
        if (local_y < RADIUS)
        {
            xy_plane[local_y][tile_x]                       = x0[thread_x0_idx -     RADIUS * stride_y_ex];
            xy_plane[local_y + Y_BLK_SIZE + RADIUS][tile_x] = x0[thread_x0_idx + Y_BLK_SIZE * stride_y_ex];
        }
        
        if (local_x < RADIUS)
        {
            xy_plane[tile_y][local_x]                       = x0[thread_x0_idx - RADIUS];
            xy_plane[tile_y][local_x + X_BLK_SIZE + RADIUS] = x0[thread_x0_idx + X_BLK_SIZE];
        }
        
        current_x0_z = z_axis_buff[RADIUS];
        xy_plane[tile_y][tile_x] = current_x0_z;
        __syncthreads();
        
        // 3. Stencil calculation
        double value = coef0 * current_x0_z;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            value += cu_coef[r] * (z_axis_buff[RADIUS + r]        + z_axis_buff[RADIUS - r] 
                                   + xy_plane[tile_y + r][tile_x] + xy_plane[tile_y - r][tile_x] 
                                   + xy_plane[tile_y][tile_x + r] + xy_plane[tile_y][tile_x - r]);
        }
        
        // 4. Store the output value
        if (valid_x1) x1[thread_x1_idx] = value;
        thread_x1_idx += stride_z;
        thread_x0_idx += stride_z_ex;
    }
}

__global__ void copy_inner_x_kernel(
    const int nx, const int ny, const int nz, 
    const double *x, double *x_ex
)
{
    const int nx_ex = nx + 2 * RADIUS;
    const int ny_ex = ny + 2 * RADIUS;
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const int ip = i + RADIUS;
    const int jp = j + RADIUS;
    const int kp = k + RADIUS;
    
    const int idx    = k  * nx    * ny    + j  * nx    + i;
    const int idx_ex = kp * nx_ex * ny_ex + jp * nx_ex + ip;
    if (i < nx || j < ny || k < nz) x_ex[idx_ex] = x[idx];
}

__global__ void cu_daxpy(const int length, const double alpha, const double *x, double *y)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;
    for (int i = tid; i < length; i += nthreads)
        y[i] += alpha * x[i];
}

__global__ void cu_dxpay(const int length, const double alpha, const double *x, double *y)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;
    for (int i = tid; i < length; i += nthreads)
        y[i] = alpha * y[i] + x[i];
}

__global__ void cu_dcopy(const int length, const double *src, double *dst)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;
    for (int i = tid; i < length; i += nthreads)
        dst[i] = src[i];
}

__global__ void cu_ddot(const int length, const double *x, const double *y, double *res)
{
    __shared__ double tmp[512];
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;
    
    double local_sum = 0.0;
    for (int i = global_tid; i < length; i += nthreads)
        local_sum += x[i] * y[i];
    tmp[tid] = local_sum;
    __syncthreads();
    
    if (tid < 256) tmp[tid] += tmp[tid + 256];  __syncthreads();
    if (tid < 128) tmp[tid] += tmp[tid + 128];  __syncthreads();
    if (tid <  64) tmp[tid] += tmp[tid +  64];  __syncthreads();
    if (tid <  32) tmp[tid] += tmp[tid +  32];  __syncthreads();
    if (tid <  16) tmp[tid] += tmp[tid +  16];  __syncthreads();
    if (tid <   8) tmp[tid] += tmp[tid +   8];  __syncthreads();
    if (tid <   4) tmp[tid] += tmp[tid +   4];  __syncthreads();
    if (tid <   2) tmp[tid] += tmp[tid +   2];  __syncthreads();
    if (tid == 0) 
    {
        tmp[0] += tmp[1];
        atomicAdd(res, tmp[0]);
    }
}

__global__ void cu_CG(
    const int nx, const int ny, const int nz, 
    const double res_tol, const int max_iter, const double *b, 
    double *x_ex, double *x, double *CG_buff
)
{
    const int n   = nx * ny * nz;
    double *r     = CG_buff + n * 0;
    double *p     = CG_buff + n * 1;
    double *s     = CG_buff + n * 2;
    double *ddot  = CG_buff + n * 3;
    
    dim3 cpy_grid, cpy_block, fd_grid, fd_block, vec_grid;
    cpy_block.x = 16;
    cpy_block.y = 4;
    cpy_block.z = 4;
    cpy_grid.x  = (nx + 15) / 16;
    cpy_grid.y  = (ny +  3) /  4;
    cpy_grid.z  = (nz +  3) /  4;
    fd_block.x  = X_BLK_SIZE;
    fd_block.y  = Y_BLK_SIZE;
    fd_grid.x   = (nx + X_BLK_SIZE - 1) / X_BLK_SIZE;
    fd_grid.y   = (ny + Y_BLK_SIZE - 1) / Y_BLK_SIZE;
    vec_grid.x  = (n + 511) / 512;
    
    copy_inner_x_kernel<<<cpy_grid, cpy_block>>>(nx, ny, nz, x, x_ex);
    cudaDeviceSynchronize();
    stencil_3axis_cuda_kernel<<<fd_grid, fd_block>>>(nx, ny, nz, x_ex, r);
    cudaDeviceSynchronize();
    cu_dxpay<<<vec_grid, 512>>>(n, -1.0, b, r);  // r = b - A * x;
    cudaDeviceSynchronize();
    cu_dcopy<<<vec_grid, 512>>>(n, r, p);        // p = r;
    cudaDeviceSynchronize();
    
    ddot[0] = 0.0;
    cu_ddot<<<vec_grid, 512>>>(n, r, r, ddot);
    cudaDeviceSynchronize();
    double r2 = ddot[0];
    double r_l2norm = sqrt(r2);
    
    ddot[0] = 0.0;
    cu_ddot<<<vec_grid, 512>>>(n, b, b, ddot);
    cudaDeviceSynchronize();
    double b2 = ddot[0];
    double b_l2norm = sqrt(b2);
    double res_stop = b_l2norm * res_tol;
    
    double r2_old, alpha, beta;
    int iter_cnt, converged = 0;
    for (iter_cnt = 1; iter_cnt < max_iter; iter_cnt++)
    {
        // (1) s = A * p;
        copy_inner_x_kernel<<<cpy_grid, cpy_block>>>(nx, ny, nz, p, x_ex);
        cudaDeviceSynchronize();
        stencil_3axis_cuda_kernel<<<fd_grid, fd_block>>>(nx, ny, nz, x_ex, s);
        cudaDeviceSynchronize();
        
        // (2) alpha = r2 / (p' * s);
        ddot[0] = 0.0;
        cu_ddot<<<vec_grid, 512>>>(n, p, s, ddot);
        cudaDeviceSynchronize();
        alpha = r2 / ddot[0];
        
        // (3) x = x + alpha * p;
        cu_daxpy<<<vec_grid, 512>>>(n,  alpha, p, x);
        cudaDeviceSynchronize();
        
        // (4) r = r - alpha * s;
        cu_daxpy<<<vec_grid, 512>>>(n, -alpha, s, r);
        cudaDeviceSynchronize();
        
        // (5) r2_old = r2; r2 = r' * r;
        r2_old = r2;
        ddot[0] = 0.0;
        cu_ddot<<<vec_grid, 512>>>(n, r, r, ddot);
        cudaDeviceSynchronize();
        r2 = ddot[0];
        r_l2norm = sqrt(r2);
        
        // (6) beta = r2 / r2_old; p = r + beta * p;
        beta = r2 / r2_old;
        cu_dxpay<<<vec_grid, 512>>>(n, beta, r, p);
        cudaDeviceSynchronize();
        
        if (r_l2norm < res_stop) 
        {
            printf("CPU CG converged, iteration = %d, ", iter_cnt);
            converged = 1;
            break;
        }
    }
    if (converged == 0)
        printf("GPU CG stopped after %d iterations, ", iter_cnt);
}
