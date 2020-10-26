#ifndef __SYCL_UTILS_HPP__
#define __SYCL_UTILS_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>

#include <iostream>

#include "CL/sycl.hpp"
namespace sycl = cl::sycl;

static double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

#endif