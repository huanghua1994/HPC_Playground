#include "../common/sycl_utils.hpp"
#include "mpi_dev_mem.h"

void fill_int_array_kernel(sycl::queue &q, const int vec_len, const int my_rank, int *d_vec)
{
    size_t global_size = static_cast<size_t>(vec_len);
    size_t local_size  = 64;
    sycl::range global_range {global_size};
    sycl::range local_range  {local_size};

    q.submit( [&](sycl::handler &h) {
        h.parallel_for<class fill_int_array> (
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> item)
        {
            int i = item.get_global_id(0);
            d_vec[i] = my_rank;
        });
    });
}

int main(int argc, char **argv)
{
    int n_proc, my_rank;
    MPI_Init_wrapper(&argc, &argv);
    MPI_Comm_world_size_rank(&n_proc, &my_rank);
    int next_rank = (my_rank + 1) % n_proc;
    int prev_rank = (my_rank == 0) ? (n_proc - 1) : my_rank - 1;

    int vec_len = 1024;
    if (argc >= 2) vec_len = atoi(argv[1]);
    if (vec_len < 1) vec_len = 1024;
    size_t vec_bytes = sizeof(int) * static_cast<size_t>(vec_len);
    if (my_rank == 0) printf("SYCL + MPI test, vector length = %d\n", vec_len);

    sycl::queue q(sycl::default_selector{});
    if (my_rank == 0) std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    try
    {
        char dev_name[128];
        int dev_name_len = q.get_device().get_info<sycl::info::device::name>().length();
        memcpy(dev_name, q.get_device().get_info<sycl::info::device::name>().c_str(), dev_name_len);
        dev_name[dev_name_len] = 0;
        printf("Rank %2d SYCL device: %s\n", my_rank, dev_name);
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }

    int *h_vec = static_cast<int *>(malloc(vec_bytes));
    int *d_vec0, *d_vec1, *d_vec2;
    try 
    {
        // Weird: if we use malloc_device here, MPI_test_dev_mem_put() will need to 
        // sleep 1 ms or longer to make it correct. Or, we can use malloc_shared 
        // here, but who will manage the data movement, SYCL runtime or MPI??
        d_vec0 = static_cast<int *>(sycl::malloc_device(vec_bytes, q));
        d_vec1 = static_cast<int *>(sycl::malloc_device(vec_bytes, q));
        d_vec2 = static_cast<int *>(sycl::malloc_device(vec_bytes, q));
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }
    
    // Initialize local host buffer and copy it to device buffer
    for (int i = 0; i < vec_len; i++) h_vec[i] = my_rank;
    try 
    {
        //q.memcpy(d_vec0, h_vec, vec_bytes);
        fill_int_array_kernel(q, vec_len, my_rank, d_vec0);
        q.wait();
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }

    // Copy device data received by MPI_Recv to host buffer and check
    MPI_test_dev_mem_recv(n_proc, my_rank, vec_len, d_vec0, d_vec1);
    try 
    {
        q.memcpy(h_vec, d_vec1, vec_bytes);
        q.wait();
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }
    int n_error = 0;
    for (int i = 0; i < vec_len; i++) 
        if (h_vec[i] != prev_rank) n_error++;
    printf("Rank %2d MPI_Recv has %d error (s)\n", my_rank, n_error);

    // Copy device data received by MPI_Get to host buffer and check
    MPI_test_dev_mem_put(n_proc, my_rank, vec_len, d_vec0, d_vec2);
    try 
    {
        q.memcpy(h_vec, d_vec2, vec_bytes);
        q.wait();
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }
    n_error = 0;
    for (int i = 0; i < vec_len; i++) 
        if (h_vec[i] != next_rank) n_error++;
    printf("Rank %2d MPI_Put  has %d error (s)\n", my_rank, n_error);

    // Remember to free the memory!
    try
    {
        sycl::free(d_vec0, q);
        sycl::free(d_vec1, q);
        sycl::free(d_vec2, q);
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }
    free(h_vec);

    MPI_Finalize_wrapper();
    return 0;
}