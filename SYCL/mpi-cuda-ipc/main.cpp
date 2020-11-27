#include "../common/sycl_utils.hpp"
#include "mpi_proxy.h"
#include "cuda_proxy.h"

void fill_int_array_kernel(sycl::queue &q, const int vec_len, const int val, int *d_vec)
{
    size_t global_size = static_cast<size_t>(vec_len);
    size_t local_size  = 64;
    global_size = (global_size + local_size - 1) / local_size * local_size;
    sycl::range global_range {global_size};
    sycl::range local_range  {local_size};

    q.submit( [&](sycl::handler &h) {
        h.parallel_for<class fill_int_array> (
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> item)
        {
            int i = item.get_global_id(0);
            d_vec[i] = val;
        });
    });
}

sycl::device select_device(sycl::device &sample_device)
{
    sycl::platform target_platform = sample_device.get_platform();
    auto platform_devices = target_platform.get_devices();
    int n_deivce    = platform_devices.size();
    int shm_my_rank = MPI_proxy_get_local_rank_env();
    int my_dev_id   = shm_my_rank % n_deivce;
    return platform_devices[my_dev_id];
}

int main(int argc, char **argv)
{
    int n_proc, my_rank, proc_name_len;
    char proc_name[1024];
    MPI_proxy_init(&argc, &argv);
    MPI_proxy_comm_size(NULL, &n_proc);
    MPI_proxy_comm_rank(NULL, &my_rank);
    MPI_proxy_get_processor_name(&proc_name[0], &proc_name_len);
    proc_name[proc_name_len] = 0;
    int next_rank = (my_rank + 1) % n_proc;
    int prev_rank = (my_rank == 0) ? (n_proc - 1) : my_rank - 1;

    int vec_len = 1024;
    if (argc >= 2) vec_len = atoi(argv[1]);
    if (vec_len < 1) vec_len = 1024;
    size_t vec_bytes = sizeof(int) * static_cast<size_t>(vec_len);
    if (my_rank == 0) 
    {
        printf("SYCL + MPI test, vector length = %d\n", vec_len);
        fflush(stdout);
    }

    sycl::device sample_device(sycl::gpu_selector{});
    sycl::device target_device = select_device(sample_device);
    sycl::queue q(target_device);

    char dev_name[128];
    int dev_name_len = q.get_device().get_info<sycl::info::device::name>().length();
    memcpy(dev_name, q.get_device().get_info<sycl::info::device::name>().c_str(), dev_name_len);
    dev_name[dev_name_len] = 0;
    printf("Rank %2d: node %s, SYCL device %s\n", my_rank, proc_name, dev_name);
    fflush(stdout);

    int *h_vec = static_cast<int *>(malloc(vec_bytes));
    int *d_vec = static_cast<int *>(sycl::malloc_device(vec_bytes, q));
    fill_int_array_kernel(q, vec_len, my_rank, d_vec);
    q.wait();

    int cu_device, cu_handle_bytes;
    void *cu_mem_handle;
    cu_device = cuda_get_device();
    cuda_get_ipc_mem_handle(d_vec, &cu_handle_bytes, &cu_mem_handle);

    void *mpi_char, *mpi_int;
    int *cu_devices = static_cast<int *>(malloc(sizeof(int) * n_proc));
    char *cu_mem_handles = static_cast<char *>(malloc(sizeof(char) * n_proc * cu_handle_bytes));
    MPI_proxy_dtype_char(&mpi_char);
    MPI_proxy_dtype_int(&mpi_int);
    MPI_proxy_allgather(&cu_device, 1, mpi_int, cu_devices, 1, mpi_int, NULL);
    MPI_proxy_allgather(cu_mem_handle, cu_handle_bytes, mpi_char, cu_mem_handles, cu_handle_bytes, mpi_char, NULL);
    
    void *dst_dptr, *dst_handle;
    int can_p2p = cuda_check_dev_p2p(cu_device, cu_devices[next_rank]);
    dst_handle = static_cast<void *>(cu_mem_handles + next_rank * cu_handle_bytes);
    cuda_open_ipc_mem_handle(&dst_dptr, dst_handle);
    int *dst_d_vec = static_cast<int *>(dst_dptr);
    printf("Rank %2d open IPC memory handle done\n", my_rank);
    fflush(stdout);
    MPI_proxy_barrier(NULL);

    fill_int_array_kernel(q, vec_len, my_rank, dst_d_vec);
    q.wait();
    printf("Rank %2d kernel IPC access done\n", my_rank);
    fflush(stdout);
    MPI_proxy_barrier(NULL);

    cuda_close_ipc_mem_handle(dst_dptr);
    printf("Rank %2d close IPC memory handle done\n", my_rank);
    fflush(stdout);
    MPI_proxy_barrier(NULL);

    q.memcpy(h_vec, d_vec, vec_bytes);
    q.wait();
    printf("Rank %2d vec[0] = %d (expected %d)\n", my_rank, h_vec[0], prev_rank);
    fflush(stdout);
    MPI_proxy_barrier(NULL);

    sycl::free(d_vec, q);
    free(h_vec);

    MPI_proxy_finalize();
    return 0;
}