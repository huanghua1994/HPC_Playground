#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "stencil_3axis_ref.h"
#include "stencil_3axis_cuda.h"

int nx = 1024;
int ny = 1024;
int nz = 1024;
int radius = 6;
int ntest  = 5;

static void print_params(int argc, char **argv)
{
    if (argc >= 2) nx = atoi(argv[1]);
    if (argc >= 3) ny = atoi(argv[2]);
    if (argc >= 4) nz = atoi(argv[3]);
    if (nx < 0 || nx > 1024) nx = 128;
    if (ny < 0 || ny > 1024) ny = 128;
    if (nz < 0 || nz > 1024) nz = 128;

    int nthreads = omp_get_max_threads();
    printf("Test parameters:\n");
    printf("  Domain size x,y,z = %d * %d * %d\n", nx, ny, nz);
    printf("  Number of threads = %d\n", nthreads);
    printf("  Test repeats      = %d\n", ntest);
    printf("\n");
}

static void init_arrays(double **stencil_coefs, double **x0, double **x1_ref, double **x1)
{
    // Initialize stencil coefficients
    double *_stencil_coefs = (double *) malloc(sizeof(double) * (radius + 1));
    for (int i = 0; i <= radius; i++)
        _stencil_coefs[i] = (double) (radius - i);
    
    // For convenience, extend the input domain by 2 * radius on each dimension
    // For convenience, we allocate the same space for the input and output domains
    size_t ext_domain_size, ext_domain_memsize; 
    ext_domain_size  = (size_t) (nx + 2 * radius);
    ext_domain_size *= (size_t) (ny + 2 * radius);
    ext_domain_size *= (size_t) (nz + 2 * radius);
    ext_domain_memsize = ext_domain_size * sizeof(double);
    double *_x0     = (double*) malloc(ext_domain_memsize);
    double *_x1     = (double*) malloc(ext_domain_memsize);
    double *_x1_ref = (double*) malloc(ext_domain_memsize);
    assert(_x0 != NULL && _x1 != NULL && _x1_ref != NULL);
    printf("Test domain memory size = %.2lf (MB)\n", (double) ext_domain_memsize / 1048576.0);
    
    // Initialize the values in the domain
    int stride_y =  nx + 2 * radius;
    int stride_z = (nx + 2 * radius) * (ny + 2 * radius);
    #pragma omp parallel for 
    for (size_t i = 0; i < ext_domain_size; i++) _x0[i] = 0.0;
    #pragma omp parallel for
    for (int z = 0; z < nz; z++)
    {
        int iz = z + radius;
        for (int y = 0; y < ny; y++)
        {
            int iy = y + radius;
            int offset_zy = iz * stride_z + iy * stride_y;
            for (int x = 0; x < nx; x++)
            {
                int ix = x + radius;
                _x0[offset_zy + ix] = (double) ((x + y + z) % 1919 + 1);
            }
        }
    }

    *stencil_coefs = _stencil_coefs;
    *x0 = _x0;
    *x1 = _x1;
    *x1_ref = _x1_ref;
}

static void test_kernel_3axis(double *stencil_coefs, double *x0, double *x1_ref, double *x1)
{
    double st, et, ut, max_err = 0.0;
    double GPoints = nx * ny * nz / 1000000000.0;
    double GFlops  = GPoints * (7.0 * radius + 1);
    
    printf("\nTest target: 3-axis stencil, radius = %d (%d points stencil)\n\n", radius, radius * 6 + 1);

    // Use reference kernel to get reference result
    for (int i = 0; i < ntest; i++)
    {
        st = omp_get_wtime();
        stencil_3axis_ref(x0, radius, stencil_coefs, nx, ny, nz, x1_ref);
        et = omp_get_wtime();
        ut = et - st;
        printf("Reference kernel finished, used time = %lf (ms), %lf GPoint/s, %lf GFlops\n", 
               ut * 1000.0, GPoints / ut, GFlops / ut);
    }
    printf("\n");
    
    // Test optimized kernel 
    stencil_3axis_cuda(x0, radius, stencil_coefs, nx, ny, nz, x1, ntest);
    
    // Check the result
    #pragma omp parallel for reduction(max:max_err)
    for (int idx = 0; idx < nx * ny * nz; idx++)
    {
        double err = x1[idx] - x1_ref[idx];
        double rel_err = fabs(err / x1_ref[idx]);
        if (rel_err > max_err) max_err = rel_err;
    }
    printf("Max relative error in results = %e, ", max_err);
    if (max_err < 1e-12) printf("PASSED\n");
    else                 printf("FAILED\n");
}

int main(int argc, char **argv)
{
    print_params(argc, argv);
    
    double *stencil_coefs, *x0, *x1, *x1_ref;

    init_arrays(&stencil_coefs, &x0, &x1_ref, &x1);
    
    test_kernel_3axis(stencil_coefs, x0, x1_ref, x1);
    
    free(x0);
    free(x1);
    free(x1_ref);
    free(stencil_coefs);
}