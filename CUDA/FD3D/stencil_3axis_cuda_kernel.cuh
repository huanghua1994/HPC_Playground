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
    const double coef0 = cu_coef[0] * cu_coef[0] * cu_coef[0];
    
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
            xy_plane[local_y][tile_x]                       = x0[thread_x0_idx - RADIUS * stride_y_ex];
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
        __syncthreads();
        
        // 4. Store the output value
        if (valid_x1) x1[thread_x1_idx] = value;
        thread_x1_idx += stride_z;
        thread_x0_idx += stride_z_ex;
        __syncthreads();
    }
}
