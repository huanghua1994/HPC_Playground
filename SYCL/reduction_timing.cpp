// This file is received from John Pennycook <john.pennycook@intel.com>

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;
using namespace sycl::ONEAPI;

using T = float;

int main(int argc, char **argv)
{
  int n = 10 * 1024 * 1024, b = 32;
  if (argc >= 2) n = std::atoi(argv[1]);
  if (argc >= 3) b = std::atoi(argv[2]);
  if (n < 0) n = 10 * 1024 * 1024;
  if (b < 0) b = 32;
  int n_blk = (n + b - 1) / b;
  n = n_blk * b;
  
  std::cout << "n = " << n << ", b = " << b << ", n_blk = " << n_blk << std::endl;

  size_t N = static_cast<size_t>(n);
  size_t B = static_cast<size_t>(b);

  queue Q{sycl::property::queue::enable_profiling{}};
  int* data = malloc_shared<int>(N, Q);
  T* sum = malloc_shared<T>(1, Q);
  std::iota(data, data + N, 1);
  *sum = 0;

  std::cout << "Selected device: " << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  sycl::event ev1 = Q.submit([&](handler& h) {
// BEGIN CODE SNIP
     h.parallel_for(
         nd_range<1>{N, B},
         reduction(sum, plus<T>()),
         [=](nd_item<1> it, auto& sum) {
           int i = it.get_global_id(0);
           sum += (T) data[i];
         });
// END CODE SNIP
   });
  ev1.wait();

  // Repeat similar kernel without reduction as a sanity check of timing
  sycl::event ev2 = Q.submit([&](handler& h) {
    h.parallel_for(
        nd_range<1>{N, B},
        [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            data[i] += 1;
        });
  });
  ev2.wait();

  auto rt1 = ev1.get_profiling_info<info::event_profiling::command_end>();
  auto rt2 = ev2.get_profiling_info<info::event_profiling::command_end>();
  rt1 -= ev1.get_profiling_info<info::event_profiling::command_start>();
  rt2 -= ev2.get_profiling_info<info::event_profiling::command_start>();
  std::cout << "Runtime with    reduction = " << rt1 << " ns" << std::endl;
  std::cout << "Runtime without reduction = " << rt2 << " ns" << std::endl;

  free(sum, Q);
  free(data, Q);
}