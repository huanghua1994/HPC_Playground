#include "../common/sycl_utils.hpp"

void atomic_add_double(double *addr, const double addend)
{
    uint64_t *addr_ = reinterpret_cast<uint64_t *>(addr);
    uint64_t expected_val, updated_val;
    double old_val, new_val;
    sycl::atomic<uint64_t> atomic_op{ sycl::global_ptr<uint64_t>{addr_} };
    do 
    {
        expected_val = atomic_op.load();
        old_val = *((double *) &expected_val);
        new_val = old_val + addend;
        updated_val  = *((uint64_t *) &new_val);
    } while (!atomic_op.compare_exchange_strong(expected_val, updated_val));
}

void atomic_add_double_test_kernel(sycl::queue &q, const int vec_len, double *a, double *sum)
{
    size_t global_size = static_cast<size_t>(vec_len);
    size_t local_size  = 256;
    sycl::range global_range {global_size};
    sycl::range local_range  {local_size};

    q.submit( [&](sycl::handler &h) {
        h.parallel_for<class stream_triad> (
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> item)
        {
            const int i = item.get_global_id(0);
            atomic_add_double(sum, a[i]);
        });
    });
}

int main(int argc, char **argv)
{
    int vec_len = 1024;
    if (argc >= 2) vec_len = atoi(argv[1]);
    if (vec_len < 1024) vec_len = 1024;
    std::cout << "SYCL atomic-add, vector length = " << vec_len << std::endl;

    size_t vec_bytes = sizeof(double) * vec_len;
    double *vec_h = static_cast<double *>(malloc(vec_bytes));
    double sum_h = 0.0, sum_d_h;
    srand48(time(NULL));
    for (int i = 0; i < vec_len; i++)
    {
        vec_h[i] = drand48() - 0.5;
        sum_h += vec_h[i];
    }
    printf("Host array sum   = %.6e\n", sum_h);


    sycl::queue q(sycl::default_selector{});
    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    double *vec_d = static_cast<double *>(sycl::malloc_device(vec_bytes, q));
    double *sum_d = static_cast<double *>(sycl::malloc_device(sizeof(double), q));
    q.memcpy(vec_d, vec_h, vec_bytes);
    q.memset(sum_d, 0, sizeof(double));
    q.wait();

    atomic_add_double_test_kernel(q, vec_len, vec_d, sum_d);
    q.wait();

    q.memcpy(&sum_d_h, sum_d, sizeof(double));
    q.wait();
    printf("Device array sum = %.6e\n", sum_d_h);
    printf("Relative error   = %.6e\n", fabs((sum_d_h - sum_h) / sum_h));

    free(vec_h);
    sycl::free(sum_d, q);
    sycl::free(vec_d, q);
    return 0;
}