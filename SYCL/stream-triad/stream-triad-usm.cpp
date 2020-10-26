#include "../common/sycl_utils.hpp"

void stream_triad_usm_kernel(
    double *x, double *y, double *z,
    const double alpha, const int vec_len, sycl::queue &q
)
{
    size_t global_size = static_cast<size_t>(vec_len);
    size_t local_size  = 256;
    sycl::range global_range {global_size};
    sycl::range local_range  {local_size};

    q.submit( [&](sycl::handler &h) {
        h.parallel_for<class stream_triad> (
        //sycl::range<1>{static_cast<size_t>(vec_len)}, [=](sycl::id<1> it)
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> it)
        {
            //const size_t i = it[0];
            const size_t i = it.get_global_id(0);
            z[i] = alpha * x[i] + y[i];
        });
    });
}

int main(int argc, char **argv)
{
    int vec_len = 1048576;
    if (argc >= 2) vec_len = atoi(argv[1]);
    if (vec_len < 1024) vec_len = 1024;
    std::cout << "SYCL stream-triad, vector length = " << vec_len << std::endl;

    // Allocate memory on host
    std::vector<double> h_x(vec_len);
    std::vector<double> h_y(vec_len);
    std::vector<double> h_z(vec_len);
    std::vector<double> d_z(vec_len);
    double alpha = 2.0;
    for (int i = 0; i < vec_len; i++)
    {
        h_x[i] = drand48();
        h_y[i] = drand48();
        h_z[i] = alpha * h_x[i] + h_y[i];
    }
    printf("Generating random vectors done\n");

    double runtime = 0.0;
    int n_test = 10;
    try 
    {
        // Create a queue on the default SYCL device and run test
        sycl::queue q(sycl::default_selector{});

        // Allocate SYCL 1D buffer
        double *x = static_cast<double *>( sycl::malloc_device<double>(static_cast<size_t>(vec_len), q) );
        double *y = static_cast<double *>( sycl::malloc_device<double>(static_cast<size_t>(vec_len), q) );
        double *z = static_cast<double *>( sycl::malloc_device<double>(static_cast<size_t>(vec_len), q) );
        size_t vec_bytes = sizeof(double) * static_cast<size_t>(vec_len);

        // Explicitly transfer data from host to device
        q.memcpy(x, h_x.data(), vec_bytes);
        q.memcpy(y, h_y.data(), vec_bytes);
        q.wait();

        // Run the kernel
        for (int k = 0; k <= n_test; k++)
        {
            double st = get_wtime_sec();
            stream_triad_usm_kernel(x, y, z, alpha, vec_len, q);
            q.wait();
            double et = get_wtime_sec();
            if (k > 0) runtime += et - st;
        }
        runtime /= static_cast<double>(n_test);

        // Explicitly transfer data back from device to host
        q.memcpy(d_z.data(), z, vec_bytes);
        q.wait();
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }

    // Check the result
    int n_error = 0;
    for (int i = 0; i < vec_len; i++)
        if (h_z[i] != d_z[i]) n_error++;
    std::cout << "There were " << n_error << " error(s)" << std::endl;

    // Compute the bandwidth
    double giga_bytes = 3.0 * static_cast<double>(vec_len * sizeof(double)) / 1024.0 / 1024.0 / 1024.0;
    double GBs = giga_bytes / runtime; 
    std::cout << "STREAM TRIAD used " << runtime << " sec, ";
    std::cout << "memory footprint = " << giga_bytes << " GBytes, ";
    std::cout << "bandwidth = " << GBs << " GB/s" << std::endl;

    return n_error;
}