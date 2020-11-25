// This file is received from John Pennycook <john.pennycook@intel.com>

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;
using namespace sycl::ONEAPI;

using T = float;

int main() {

  constexpr size_t N = 10 * 1024 * 1024;
  constexpr size_t B = 16;

  queue Q{sycl::property::queue::enable_profiling{}};
  int* data = malloc_shared<int>(N, Q);
  T* sum = malloc_shared<T>(1, Q);
  std::iota(data, data + N, 1);
  *sum = 0;

  sycl::event ev = Q.submit([&](handler& h) {
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
  ev.wait();

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

  std::cout << "reduction = " << ev.get_profiling_info<info::event_profiling::command_end>() -
      ev.get_profiling_info<info::event_profiling::command_start>() << std::endl;

  std::cout << "no reduction = " << ev2.get_profiling_info<info::event_profiling::command_end>() -
      ev2.get_profiling_info<info::event_profiling::command_start>() << std::endl;

  free(sum, Q);
  free(data, Q);
}