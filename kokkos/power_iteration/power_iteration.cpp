#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <omp.h>

typedef float DTYPE;

using ivec_view      = Kokkos::View<int*>;
using dvec_view      = Kokkos::View<DTYPE*>;
using dmat_view      = Kokkos::View<DTYPE**>;
using host_ivec_view = ivec_view::HostMirror;
using host_dvec_view = dvec_view::HostMirror;
using host_dmat_view = dmat_view::HostMirror;

using member_type = Kokkos::TeamPolicy<>::member_type;

void Kokkos_GEMV(dmat_view A, dvec_view x, dvec_view y)
{
    int m = A.extent(0);
    int n = A.extent(1);
    Kokkos::parallel_for(
        "GEMV", 
        Kokkos::TeamPolicy<>(m, Kokkos::AUTO), 
        KOKKOS_LAMBDA(const member_type &team_member)
        {
            int row = team_member.league_rank();
            DTYPE row_dot_sum = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, n),
                [=] (const int col, DTYPE &lsum) 
                {
                    lsum += A(row, col) * x(col); 
                },
                row_dot_sum
            );  // End of Kokkos::parallel_reduce
            if (team_member.team_rank() == 0) y(row) = row_dot_sum;
        }  // End of KOKKOS_LAMBDA
    );  // End of Kokkos::parallel_for
}

DTYPE Kokkos_dot(dvec_view x, dvec_view y)
{
    assert(x.extent(0) == y.extent(0));
    DTYPE res = 0.0;
    Kokkos::parallel_reduce(
        "DOT", x.extent(0), 
        KOKKOS_LAMBDA(const int i, DTYPE& lsum) { lsum += x(i) * y(i); }, res
    );
    return res;
}

void host_power_iteration(
    const int n, const DTYPE *A, DTYPE *x, DTYPE *y, 
    const DTYPE reltol, const int max_iter
)
{
    DTYPE inv_sqrt_n = 1.0 / sqrt(n);
    for (int i = 0; i < n; i++) x[i] = inv_sqrt_n;

    int iter = 0;
    DTYPE relerr = 19241112.0, y_2norm, inv_y_2norm;
    double st = omp_get_wtime();
    while (relerr > reltol && iter < max_iter)
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
        {
            DTYPE res = 0.0;
            const DTYPE *A_i = A + i * n;
            #pragma omp simd
            for (int j = 0; j < n; j++) res += A_i[j] * x[j];
            y[i] = res;
        }

        y_2norm = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:y_2norm)
        for (int i = 0; i < n; i++) y_2norm += y[i] * y[i];

        y_2norm = sqrt(y_2norm);
        inv_y_2norm = 1.0 / y_2norm;
        relerr = 0.0;

        #pragma omp parallel for schedule(static) reduction(+:relerr)
        for (int i = 0; i < n; i++)
        {
            y[i] *= inv_y_2norm;
            DTYPE diff = y[i] - x[i];
            relerr += diff * diff;
            x[i] = y[i];
        }

        relerr = sqrt(relerr);
        iter++;
    }  // End of while loop
    double et = omp_get_wtime();
    printf("OpenMP : after %d iterations relerr = %e, used time = %.3lf sec\n", iter, relerr, et - st);
}

int main(int argc, char **argv)
{
    int max_iter = 1000;
    int nx, ny, n;
    DTYPE reltol = 1e-6;
    printf("FD points on x & y direction = ");
    scanf("%d%d", &nx, &ny);
    n = nx * ny;
    //printf("Relative error tolerance = ");
    //scanf("%lf", &reltol);

    DTYPE *host_A, *host_x, *host_y;
    host_A = (DTYPE*) malloc(sizeof(DTYPE) * n * n);
    host_x = (DTYPE*) malloc(sizeof(DTYPE) * n);
    host_y = (DTYPE*) malloc(sizeof(DTYPE) * n);

    Kokkos::initialize(argc, argv);
    {
        dmat_view A("A", n, n);
        dvec_view x("x", n);
        dvec_view y("y", n);
        dvec_view z("z", n);
        host_dmat_view h_A = Kokkos::create_mirror_view(A);
        host_dvec_view h_x = Kokkos::create_mirror_view(x);

        const int stencil_order = 3;
        dvec_view stencils("stencils", stencil_order + 1);
        host_dvec_view h_stencils = Kokkos::create_mirror_view(stencils);
        h_stencils(0) = 10.0;
        h_stencils(1) = -2.0;
        h_stencils(2) = -1.0;
        h_stencils(3) = -0.5;
        Kokkos::deep_copy(stencils, h_stencils);

        // Populate dense stencil matrix, also for host
        Kokkos::parallel_for(
            "populate_dense_stencil_matrix",
            Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {nx, ny}),
            KOKKOS_LAMBDA(const int ix, const int iy)
            {
                int row = ix * ny + iy;
                for (int col = 0; col < n; col++) 
                {
                    h_A(row, col) = 0.0;
                }
                h_A(row, row) = 2.0 * h_stencils(0);
                for (int r = 1; r <= stencil_order; r++)
                {
                    #define ADD_FD_POINT_TO_MAT(ix1, iy1)   \
                    if (ix1 >= 0 && ix1 < nx &&             \
                        iy1 >= 0 && iy1 < ny)               \
                    {                                       \
                        int col = ix1 * ny + iy1;           \
                        h_A(row, col) = h_stencils(r);      \
                    }

                    int ixpr = ix + r;
                    int ixmr = ix - r;
                    int iypr = iy + r;
                    int iymr = iy - r;
                    ADD_FD_POINT_TO_MAT(ixpr, iy)
                    ADD_FD_POINT_TO_MAT(ixmr, iy)
                    ADD_FD_POINT_TO_MAT(ix, iypr)
                    ADD_FD_POINT_TO_MAT(ix, iymr)

                    #undef ADD_FD_POINT_TO_MAT
                }  // End of r loop
            }  // End of KOKKOS_LAMBDA
        );  // End of Kokkos::parallel_for
        Kokkos::deep_copy(A, h_A);

        memcpy(host_A, h_A.data(), sizeof(DTYPE) * n * n);

        // Initialize initial vector whose 2-norm == 1
        DTYPE inv_sqrt_n = 1.0 / sqrt(n);
        Kokkos::parallel_for("init_vec", n, KOKKOS_LAMBDA(const int i) { x(i) = inv_sqrt_n; });

        // Power iteration
        Kokkos::Timer timer;
        int iter = 0;
        DTYPE relerr = 19241112.0, y_2norm, inv_y_2norm;
        while (relerr > reltol && iter < max_iter)
        {
            // Calculate y = A * x
            Kokkos_GEMV(A, x, y);

            // Normalize y and calculate relative error
            y_2norm = sqrt(Kokkos_dot(y, y));
            inv_y_2norm = 1.0 / y_2norm;
            Kokkos::parallel_for(
                "VSCALE", y.extent(0), 
                KOKKOS_LAMBDA(const int i) 
                { 
                    y(i) *= inv_y_2norm;
                    z(i) = x(i) - y(i); 
                }
            );
            relerr = sqrt(Kokkos_dot(z, z));

            Kokkos::deep_copy(x, y);
            iter++;
        }  // End of while loop
        double kk_time = timer.seconds();
        printf("Kokkos : after %d iterations relerr = %e, used time = %.3lf sec\n", iter, relerr, kk_time);
    }
    Kokkos::finalize();

    host_power_iteration(n, host_A, host_x, host_y, reltol, max_iter);
    free(host_A);
    free(host_x);
    free(host_y);
    return 0;
}