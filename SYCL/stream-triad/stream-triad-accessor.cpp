#include "../common/sycl_utils.hpp"

void stream_triad_accessor_kernel(
    std::vector<double> &h_x, std::vector<double> &h_y, std::vector<double> &d_z,
    sycl::buffer<double, 1> &d_x, sycl::buffer<double, 1> &d_y, sycl::buffer<double, 1> &d_z_, 
    const double alpha, const int vec_len, sycl::queue &q
)
{
    size_t global_size = static_cast<size_t>(vec_len);
    size_t local_size  = 256;
    sycl::range global_range {global_size};
    sycl::range local_range  {local_size};

    q.submit( [&](sycl::handler &h) {
        sycl::accessor x(d_x,  h, sycl::read_only);
        sycl::accessor y(d_y,  h, sycl::read_only);
        sycl::accessor z(d_z_, h, sycl::read_write);

        h.parallel_for<class stream_triad> (
        //sycl::range<1>{static_cast<size_t>(vec_len)}, [=](sycl::id<1> it)
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> it)
        {
            //const size_t i = it[0];
            const size_t i = it.get_global_id(0);
            z[i] += alpha * x[i] + y[i];
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
        h_z[i] = drand48();
        d_z[i] = h_z[i];
        h_z[i] += alpha * h_x[i] + h_y[i];
    }
    printf("Generating random vectors done\n");

    double runtime = 0.0;
    int n_test = 10;
    try 
    {
        // Create a queue on the default SYCL device and run test
        sycl::queue q(sycl::default_selector{});

        // Allocate SYCL 1D buffer
        sycl::buffer<double, 1> d_x { h_x.data(), sycl::range<1>(h_x.size()) };
        sycl::buffer<double, 1> d_y { h_y.data(), sycl::range<1>(h_y.size()) };
        sycl::buffer<double, 1> d_z_{ d_z.data(), sycl::range<1>(d_z.size()) };

        // Run the kernel, automatically transfer data to device and back from device
        for (int k = 0; k <= n_test; k++)
        {
            double st = get_wtime_sec();
            stream_triad_accessor_kernel(
                h_x, h_y, d_z, 
                d_x, d_y, d_z_, 
                alpha, vec_len, q
            );
            q.wait();
            double et = get_wtime_sec();
            if (k > 0) runtime += et - st;
        }
        runtime /= static_cast<double>(n_test);
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
    double giga_bytes = 4.0 * static_cast<double>(vec_len * sizeof(double)) / 1024.0 / 1024.0 / 1024.0;
    double GBs = giga_bytes / runtime; 
    std::cout << "STREAM TRIAD used " << runtime << " sec, ";
    std::cout << "memory footprint = " << giga_bytes << " GBytes, ";
    std::cout << "bandwidth = " << GBs << " GB/s" << std::endl;

    return n_error;
}