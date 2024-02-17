#include <iostream>
#include <cmath>
using namespace std;

#include "structured/StructuredMatrix.hpp"
#include "iterative/IterativeSolversMPI.hpp"
using namespace strumpack;

int npt, dim;
double l, mu, _2l2, sqrt3_l;
double *coord = NULL;

template<typename scalar_t> void
print_info(const MPIComm& comm,
           const structured::StructuredMatrix<scalar_t>* H,
           const structured::StructuredOptions<scalar_t>& opts) 
{
    if (comm.is_root())
    {
        cout << get_name(opts.type()) << endl
             << "  - total_nonzeros(H) = " << H->nonzeros() << endl
             << "  - total_memory(H) = " << H->memory() / 1e6 << " MByte" << endl
             << "  - maximum_rank(H) = " << H->rank() << endl;
    }
}

// HODLR does not guarantee SPD but the reviewer asked us to try PCG
void PCG(
    const MPIComm &comm, const BLACSGrid &grid, 
    const structured::StructuredMatrix<double> *H,
    const DistributedMatrix<double> &A, const DistributedMatrix<double> &b,
    const int n, const double PCG_tol, const int max_iter
)
{
    double st = MPI_Wtime();

    // x, r, z, p, s have the same partitioning
    DistributedMatrix<double> x(&grid, n, 1);
    DistributedMatrix<double> r(&grid, n, 1);
    DistributedMatrix<double> z(&grid, n, 1);
    DistributedMatrix<double> p(&grid, n, 1);
    DistributedMatrix<double> s(&grid, n, 1);
    double *x_local = x.data();
    double *r_local = r.data();
    double *z_local = z.data();
    double *p_local = p.data();
    double *s_local = s.data();
    int srow = 0, erow = 0, scol = 0, ecol = 0;
    z.lranges(srow, erow, scol, ecol);
    int nrow = erow - srow;
    int ncol = ecol - scol;

    printf("Rank %d: srow, erow, scol, ecol = %d, %d, %d, %d\n", comm.rank(), srow, erow, scol, ecol);
    if (ncol > 0)
        for (int i = 0; i < nrow; i++) x_local[i] = 0;
    comm.barrier();

    // r = b - A * x
    A.mult(Trans::N, x, r);
    r.scale_and_add(-1.0, b);

    double r_2norm = r.normF();
    double b_2norm = b.normF();
    double stop_2norm = b_2norm * PCG_tol; 
    if (comm.is_root())
    {
        printf(
            "PCG:\nmax_iter = %d\n||b||_2 = %e\ninitial  ||r||_2 = %e\nstopping ||r||_2 = %e\n", 
            max_iter, b_2norm, r_2norm, stop_2norm
        );
        printf("Iter    relres\n");
    }

    int iter = 0;
    double alpha, beta, rho0, tmp, rho = 1.0;
    while (iter < max_iter && r_2norm > stop_2norm)
    {
        // z = M \ r;
        z = r;
        H->solve(z);

        // rho0 = rho;
        // rho  = r' * z;
        // beta = rho / rho0;
        rho0 = rho;
        rho  = 0;
        if (ncol > 0)
            for (int i = 0; i < nrow; i++) rho += r_local[i] * z_local[i];
        MPI_Allreduce(MPI_IN_PLACE, &rho, 1, MPI_DOUBLE, MPI_SUM, comm.comm());
        beta = rho / rho0;

        // p = z + beta * p; or p = z;
        if (iter == 0) p = z;
        else p.scale_and_add(beta, z);

        // s = A * p;
        A.mult(Trans::N, p, s);

        // alpha = rho / (p' * s);
        tmp = 0;
        if (ncol > 0)
            for (int i = 0; i < nrow; i++) tmp += p_local[i] * s_local[i];
        MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_SUM, comm.comm());
        alpha = rho / tmp;

        // x = x + alpha * p;
        // r = r - alpha * s;
        r_2norm = 0;
        if (ncol > 0)
        {
            for (int i = 0; i < nrow; i++) 
            {
                x_local[i] += alpha * p_local[i];
                r_local[i] -= alpha * p_local[i];
                r_2norm += r_local[i] * r_local[i];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &r_2norm, 1, MPI_DOUBLE, MPI_SUM, comm.comm());
        r_2norm = std::sqrt(r_2norm);

        iter++;
        if (comm.is_root()) 
        {
            printf("%4d    %e\n", iter, r_2norm / b_2norm);
            fflush(stdout);
        }
    }

    double et = MPI_Wtime();
    if (comm.is_root())
    {
        printf("PCG performed %d iterations, time = %.2f s\n", iter, et - st);
        fflush(stdout);
    }
}

int main(int argc, char *argv[]) 
{
    int leaf_size = 200, max_iter = 500;
    double PCG_tol = 1e-4;

    // C++ wrapper around an MPI_Comm, defaults to MPI_COMM_WORLD
    MPIComm world;

    // We need at least MPI_THREADS_FUNNELED support
    int thread_level, my_rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (thread_level != MPI_THREAD_FUNNELED && my_rank == 0)
        printf("MPI implementation does not support MPI_THREAD_FUNNELED\n");

    if (argc < 5)
    {
        if (world.is_root())
        {
            printf("Usage: %s kid coord_txt l mu leaf_size PCG_tol max_iter\n", argv[0]);
            printf("kid: Kernel ID, 0 for Gaussian, 1 for Matern 3/2\n");
            printf("l, mu: kernel parameter and diagonal shift\n");
            printf("Optional: leaf_size (default 200), PCG_tol (default 1e-4), max_iter (default 500)\n");
        }
        return 255;
    }
    int kid = atoi(argv[1]);
    l  = atof(argv[3]);
    mu = atof(argv[4]);
    if (argc >= 6) leaf_size = atoi(argv[5]);
    if (argc >= 7) PCG_tol   = atof(argv[6]);
    if (argc >= 8) max_iter  = atoi(argv[7]);
    _2l2 = l * l * 2.0;
    sqrt3_l = sqrt(3.0) / l;
    if (world.is_root())
    {
        if (kid == 0) printf("Kernel: exp(-|x-y|^2 / (2 * l^2))\n");
        else          printf("Kernel: (1 + k) * exp(-k), k = sqrt(3) / l * |x-y|\n");
        printf("l, mu, PCG_tol, max_iter = %.3f, %.3f, %e, %d\n", l, mu, PCG_tol, max_iter);
        fflush(stdout);
    }

    if (world.is_root())
    {
        FILE *inf = fopen(argv[2], "r");
        fscanf(inf, "%d %d", &npt, &dim);

        printf("npt, dim   = %d, %d\n", npt, dim);
        fflush(stdout);
        coord = (double *) malloc(sizeof(double) * npt * dim);
        for (int i = 0; i < npt * dim; i++) fscanf(inf, "%lf", coord + i);
        fclose(inf);
        printf("\nRead point coordinates from input file done\n");
        fflush(stdout);
    }
    MPI_Bcast(&npt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0) coord = (double *) malloc(sizeof(double) * npt * dim);
    MPI_Bcast(coord, npt * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Close this scope to destroy everything before calling MPI_Finalize and Cblacs_exit
    {
        // Define an options object, set to the default options.
        structured::StructuredOptions<double> options;
        // Suppress some output
        options.set_verbose(false);
        // Parse options passed on the command line, run with --help to see more
        options.set_from_command_line(argc, argv);

        // Create a 2D processor grid, as used in ScaLAPACK. This is only
        // needed if you will use 2d block-cyclic input/output.
        BLACSGrid grid(world);

        // Define the matrix through a routine to compute individual elements of the matrix
        auto MyGaussianKernel = [](const int i, const int j)
        {
            if (i == j) return (1 + mu);
            else
            {
                double R2 = 0;
                double *x_i = coord + ((ptrdiff_t) i * (ptrdiff_t) dim);
                double *x_j = coord + ((ptrdiff_t) j * (ptrdiff_t) dim);
                for (int k = 0; k < dim; k++)
                {
                    double diff = x_i[k] - x_j[k];
                    R2 += diff * diff;
                }
                return exp(-R2 / _2l2);
            }
        };
        auto MyMatern32Kernel = [](const int i, const int j)
        {
            if (i == j) return (1 + mu);
            else
            {
                double R2 = 0, R = 0;
                double *x_i = coord + ((ptrdiff_t) i * (ptrdiff_t) dim);
                double *x_j = coord + ((ptrdiff_t) j * (ptrdiff_t) dim);
                for (int k = 0; k < dim; k++)
                {
                    double diff = x_i[k] - x_j[k];
                    R2 += diff * diff;
                }
                R = sqrt(R2);
                return (1 + sqrt3_l * R) * exp(-sqrt3_l * R);
            }
        };
        auto target_kernel = (kid == 0) ? MyGaussianKernel : MyMatern32Kernel;

        srand(my_rank);
        auto rand_matrix_elem = [](const int i, const int j)
        {
            double val = ((double) rand() / (double) RAND_MAX);
            return (2.0 * val - 1.0);
        };

        double st, et;

        // Set structured matrix options
        options.set_rel_tol(1e-2);  // The reviewer says 1e-2 is enough for preconditioner
        options.set_leaf_size(leaf_size);
        options.set_type(structured::Type::HODLR);
        options.set_max_rank(npt);

        // Build a StructuredMatrix
        st = MPI_Wtime();
        auto H = structured::construct_from_elements<double>
            (world, &grid, npt, npt, target_kernel, options);
        et = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Build StructuredMatrix used %.3f s\n", et - st);
            fflush(stdout);
        }

        structured::StructuredMatrix<double> *H_ = H.get();
        print_info(world, H_, options);

        st = MPI_Wtime();
        H_->factor();
        et = MPI_Wtime();
        if (my_rank == 0) 
        {
            printf("StructuredMatrix factor time = %.3f\n", et - st);
            fflush(stdout);
        }

        // Create a 2DBC distributed matrix, and initialize it as a Gaussian matrix
        st = MPI_Wtime();
        DistributedMatrix<double> A2d(&grid, npt, npt, target_kernel);
        et = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Create dense kernel matrix used %.3f s\n", et - st);
            printf("matrix memory on rank 0 = %.3f MB\n", A2d.memory() / 1048576.0);
            fflush(stdout);
        }
        world.barrier();

        // RHS vector
        DistributedMatrix<double> b(&grid, npt, 1, rand_matrix_elem);

        // Solve the linear system using PCG
        PCG(world, grid, H_, A2d, b, npt, PCG_tol, max_iter);
    }  
    
    scalapack::Cblacs_exit(1);
    MPI_Finalize();
    return 0;
}
