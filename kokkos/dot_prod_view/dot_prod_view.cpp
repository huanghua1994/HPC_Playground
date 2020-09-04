#include <Kokkos_Core.hpp>
#include <cstdio>

using vector_view      = Kokkos::View<double*>;
using host_vector_view = vector_view::HostMirror;

struct init_vector_view
{
    vector_view a;
    int seed;

    init_vector_view(vector_view a_, int seed_) : a(a_), seed(seed_) {}

    using value_type = double;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const
    {
        a(i) = (double) ((i + seed) % 10);
    }
};

struct dot_prod_functor
{
    vector_view a, b;

    dot_prod_functor(vector_view a_, vector_view b_) : a(a_), b(b_) {}

    using value_type = double;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& lsum) const
    {
        lsum += a(i) * b(i);
    }
};

// The following functor only works on host
struct dot_prod_functor_host
{
    double *a, *b;

    dot_prod_functor_host(double *a_, double *b_) : a(a_), b(b_) {}

    using value_type = double;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& lsum) const
    {
        lsum += a[i] * b[i];
    }
};

int main(int argc, char* argv[]) 
{
    Kokkos::initialize(argc, argv);

    int n, seed_a, seed_b;
    printf("Vector size, seed_a, seed_b = ");
    scanf("%d%d%d", &n, &seed_a, &seed_b);

    // Get host reference result
    double *host_a = (double*) malloc(sizeof(double) * n);
    double *host_b = (double*) malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        host_a[i] = (double) ((i + seed_a) % 10);
        host_b[i] = (double) ((i + seed_b) % 10);
    }
    double host_dot_res = 0;
    for (int i = 0; i < n; i++) 
        host_dot_res += host_a[i] * host_b[i];

    // Get Kokkos result
    double kk_dot_res = 0;
    {
        vector_view vec_a("vec_a", n);
        vector_view vec_b("vec_a", n);

        // Initialize with functor
        //Kokkos::parallel_for(n, init_vector_view(vec_a, seed_a));
        //Kokkos::parallel_for(n, init_vector_view(vec_b, seed_b));

        // Copy from host arrays
        host_vector_view hva = Kokkos::create_mirror_view(vec_a);
        host_vector_view hvb = Kokkos::create_mirror_view(vec_b);
        for (int i = 0; i < n; i++)
        {
            hva(i) = host_a[i];
            hvb(i) = host_b[i];
        }
        Kokkos::deep_copy(vec_a, hva);
        Kokkos::deep_copy(vec_b, hvb);

        // Calculate the dot product
        //Kokkos::parallel_reduce(n, dot_prod_functor(vec_a, vec_b), kk_dot_res);
        // Remember to add the following line to makefile for CUDA
        // KOKKOS_CUDA_OPTIONS += "enable_lambda"
        Kokkos::parallel_reduce(
            n, 
            KOKKOS_LAMBDA(const int i, double& lsum)
            {
                lsum += vec_a(i) * vec_b(i);
            }, 
            kk_dot_res
        );
    }  // Use this scope to ensure the lifetime of vec_a and vec_b end before finalize
    
    // The following code only works on host
    #if 0
    // Kokkos::parallel_reduce(n, dot_prod_functor_host(host_a, host_b), kk_dot_res);
    Kokkos::parallel_reduce(
        n, 
        KOKKOS_LAMBDA(const int i, double& lsum)
        {
            lsum += host_a[i] * host_b[i];
        }, 
        kk_dot_res
    );
    #endif

    printf("Host dot result = %lf, Kokkos dot result = %lf\n", host_dot_res, kk_dot_res);

    free(host_a);
    free(host_b);

    Kokkos::finalize();
    return 0;
}
