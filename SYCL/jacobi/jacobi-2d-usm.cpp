#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <iostream>

#include "CL/sycl.hpp"
namespace sycl = cl::sycl;

#define real float
#define PI   3.1415926585
#define tol  1e-6

double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

double sycl_event_elapsed_sec(sycl::event &e)
{
    double st = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    double et = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (et - st) * 1.0e-9;
}

void init_boundaries_kernel(
    sycl::queue &q, const int grid_dim_x, const int block_dim_x,
    real *a_new, real *a, const int offset, const int nx, const int my_ny, int ny
)
{
    size_t global_size = static_cast<size_t>(grid_dim_x * block_dim_x);
    size_t local_size  = static_cast<size_t>(block_dim_x);
    sycl::range global_range {global_size};
    sycl::range local_range  {local_size};

    q.submit( [&](sycl::handler &h) {
        h.parallel_for<class init_boundaries> (
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> item)
        {
            int iy_start = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            int n_workitem = item.get_local_range(0) * item.get_group_range(0);
            for (int iy = iy_start; iy < my_ny; iy += n_workitem)
            {
                real val = sycl::sin(2.0 * PI * static_cast<real>(offset + iy) / static_cast<real>(ny - 1));
                int left_idx  = (iy + 1) * nx + 0;
                int right_idx = (iy + 1) * nx + (nx - 1);
                a[left_idx]  = val;
                a[right_idx] = val;
                a_new[left_idx]  = val;
                a_new[right_idx] = val;
            }
        });
    });
}

void jacobi_single_device_compute_kernel(
    sycl::queue &q, sycl::event &event_wait, sycl::event &event_compute, 
    sycl::range<2> &global_range, sycl::range<2> &local_range,
    real *a_new, real *a, real *l2_norm, const int nx, 
    const int iy_start, const int iy_end
)
{
    event_compute = q.submit( [&](sycl::handler &h) {
        h.depends_on(event_wait);
        h.parallel_for<class jacobi_single_device> (
        sycl::nd_range{global_range, local_range}, 
        sycl::ONEAPI::reduction(l2_norm, std::plus<>()), 
        [=](sycl::nd_item<2> item, auto &l2_norm_sum)
        {
            int iy = item.get_global_id(1) + iy_start;
            int ix = item.get_global_id(0) + 1;
            real local_l2_norm = 0.0;

            // Interior part, update value and calculate residual
            if ((iy < iy_end) && (ix < nx - 1))
            {
                int center_idx = iy * nx + ix;
                real new_val = 0.25 * (a[center_idx + 1]  + a[center_idx - 1] + 
                                       a[center_idx + nx] + a[center_idx - nx]);
                a_new[center_idx] = new_val;
                real residue  = new_val - a[center_idx];
                local_l2_norm = residue * residue;
            }

            item.barrier();

            // Accumulate local residual to global residual
            real group_sum = sycl::ONEAPI::reduce(item.get_group(), local_l2_norm, std::plus<>());
            if (item.get_local_linear_id() == 0) l2_norm_sum += group_sum;
        });
    });
}

void jacobi_single_device_boundary_kernel(
    sycl::queue &q, sycl::event &event_wait, sycl::event &event_boundary, 
    real *a_new, const int nx, const int iy_start, const int iy_end
)
{
    int local_size_x  = 256;
    int global_size_x = (nx + local_size_x - 1) / local_size_x;
    global_size_x *= local_size_x;
    sycl::range<1> local_range  {static_cast<size_t>(local_size_x)};
    sycl::range<1> global_range {static_cast<size_t>(global_size_x)};

    event_boundary = q.submit( [&](sycl::handler &h) {
        h.depends_on(event_wait);
        h.parallel_for<class jacobi_single_device_boundary> (
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> item)
        {
            int ix = item.get_global_id(0) + 1;
            if (ix < nx - 1)
            {
                a_new[ iy_end        * nx + ix] = a_new[ iy_start    * nx + ix];
                a_new[(iy_start - 1) * nx + ix] = a_new[(iy_end - 1) * nx + ix];
            }
        });
    });
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        printf("Usage: %s <nx> <ny> <block_x> <block_y>\n", argv[0]);
        return 255;
    }

    sycl::property_list prop_list = {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()};
    sycl::queue q(sycl::default_selector{}, prop_list);
    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Create another queue halo_q using the same device & context for halo exchange,
    // so the device memory allocated on q can be used on halo_q
    sycl::context target_context = q.get_context();
    sycl::device  target_device  = q.get_device();
    sycl::queue   reset_l2_norm_q(target_context, target_device, prop_list);

    int nx    = atoi(argv[1]);
    int ny    = atoi(argv[2]);
    int blk_x = atoi(argv[3]);
    int blk_y = atoi(argv[4]);
    
    int iter_max = 1000;
    int nccheck  = 1;
    printf(
        "Single device Jacobi relaxation: %d iterations on %d x %d mesh with "
        "norm check every %d iterations\n", iter_max, ny, nx, nccheck
    );

    int iy_start = 1;
    int iy_end   = ny - 1;

    size_t mat_bytes = sizeof(real) * static_cast<size_t>(nx * ny);
    real *a     = static_cast<real *>(sycl::malloc_device(mat_bytes, q));
    real *a_new = static_cast<real *>(sycl::malloc_device(mat_bytes, q));
    q.memset(a,     0, mat_bytes);
    q.memset(a_new, 0, mat_bytes);
    q.wait();

    // Set Dirichlet boundary conditions on left and right boarder
    init_boundaries_kernel(q, ny / 128 + 1, 128, a_new, a, 0, nx, ny - 2, ny - 2);
    q.wait();

    sycl::event event_l2_norm_copy[2], event_l2_norm_reset[2];
    sycl::event event_compute, event_boundary;
    real l2_norms[2];
    real *l2_norms_h = static_cast<real *>(sycl::malloc_host  (sizeof(real) * 2, q));
    real *l2_norms_d = static_cast<real *>(sycl::malloc_device(sizeof(real) * (iter_max + 2), q));
    l2_norms[0]   = 1.0;
    l2_norms[1]   = 1.0;
    l2_norms_h[0] = 1.0;
    l2_norms_h[1] = 1.0;
    event_l2_norm_copy[0]  = q.memcpy(&l2_norms_d[0], &l2_norms_h[0], sizeof(real));
    event_l2_norm_copy[1]  = q.memcpy(&l2_norms_d[1], &l2_norms_h[1], sizeof(real));
    q.wait();
    event_l2_norm_reset[0] = q.memset(l2_norms_d, 0, sizeof(real) * 2);
    event_l2_norm_reset[1] = event_l2_norm_reset[0];
    event_l2_norm_reset[0].wait();
    event_l2_norm_reset[1].wait();
    q.wait();

    // Workspace size
    int glb_x = (nx + blk_x - 1) / blk_x * blk_x;
    int glb_y = (ny + blk_y - 1) / blk_y * blk_y;
    sycl::range<2> local_range  {static_cast<size_t>(blk_x), static_cast<size_t>(blk_y)};
    sycl::range<2> global_range {static_cast<size_t>(glb_x), static_cast<size_t>(glb_y)};

    int converged = 0, iter = 0;
    double compute_ms = 0.0, boundary_ms = 0.0;
    double start_t = get_wtime_sec();
    while (converged == 0 && iter < iter_max)
    {
        // On new iteration: old current vars are now previous vars, old
        // previous vars are no longer needed
        int prev = iter % 2;
        int curr = (iter + 1) % 2;

        q.memset(&l2_norms_d[curr], 0, sizeof(real));
        q.wait();

        jacobi_single_device_compute_kernel(
            q, event_l2_norm_reset[curr], event_compute, 
            global_range, local_range, 
            a_new, a, &l2_norms_d[curr], nx, iy_start, iy_end
        );

        jacobi_single_device_boundary_kernel(
            q, event_compute, event_boundary, 
            a_new, nx, iy_start, iy_end
        );

        if ((iter % nccheck) == 0 || ((iter % 100) == 0)) 
        {
            // As soon as computation is done, D2H copy L2 norm
            event_l2_norm_copy[curr] = q.submit( [&](sycl::handler &h) {
                h.depends_on(event_boundary);
                h.memcpy(&l2_norms_h[curr], &l2_norms_d[curr], sizeof(real));
            });
            
            // Ensure previous D2H copy is completed before using the data for calculation
            event_l2_norm_copy[prev].wait();

            event_compute.wait();
            event_boundary.wait();
            compute_ms  += 1000.0 * sycl_event_elapsed_sec(event_compute);
            boundary_ms += 1000.0 * sycl_event_elapsed_sec(event_boundary);

            // Calculate the L2 norm
            l2_norms[prev] = std::sqrt(l2_norms_h[prev]);
            converged = (l2_norms[prev] <= tol);

            if ((iter % 100) == 0) 
            {
                printf("%5d, %0.6f\n", iter, l2_norms[prev]);
                fflush(stdout);
            }

            // Reset L2 norm for next iteration
            l2_norms[prev] = 0.0;
            l2_norms_h[prev] = 0.0;
            event_l2_norm_reset[prev] = q.memset(&l2_norms_d[prev], 0, sizeof(real));
        }

        std::swap(a_new, a);
        iter++;
    }
    double end_t = get_wtime_sec();

    printf("%4d Jacobi iterations total time = %.3f ms\n", iter, 1000.0 * (end_t - start_t));
    printf("Jacobi compute kernel used time   = %.3f ms\n", compute_ms);
    printf("Period boundary update used time  = %.3f ms\n", boundary_ms);

    float *h_a = (float *) malloc(mat_bytes);
    float *h_a_new = (float *) malloc(mat_bytes);
    q.memcpy(h_a, a, mat_bytes);
    q.memcpy(h_a_new, a_new, mat_bytes);
    q.wait();
    float h_l2_norm = 0.0;
    for (int iy = 1; iy < ny - 1; iy++)
    {
        int idx_base = iy * nx;
        for (int ix = 1; ix < nx - 1; ix++)
        {
            float diff = h_a[idx_base + ix] - h_a_new[idx_base + ix];
            h_l2_norm += diff * diff;
        }
    }
    h_l2_norm = std::sqrt(h_l2_norm);
    free(h_a);
    free(h_a_new);
    printf("Host calculated final residual L2 norm = %0.6f\n", h_l2_norm);

    double GBs = static_cast<double>((nx - 2) * (ny - 2)) * static_cast<double>(sizeof(real));
    GBs *= static_cast<double>(iter * 5);
    GBs /= 1024.0 * 1024.0 * 1024.0;
    GBs *= 1000.0;
    GBs /= compute_ms;
    printf("Jacobi kernel compute bandwidth = %.3f GB/s\n", GBs);

    q.wait();
    sycl::free(l2_norms_h, q);
    sycl::free(l2_norms_d, q);
    sycl::free(a_new, q);
    sycl::free(a, q);

    return 0;
}