#include <iostream>
#include <cmath>
using namespace std;

#include "structured/StructuredMatrix.hpp"
#include "iterative/IterativeSolversMPI.hpp"
using namespace strumpack;

int npt, dim;
double l, mu, l2, _2l2, sqrt3_l;
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


int main(int argc, char *argv[]) 
{
    // 200 is the default leaf size used by STRUMPACK
    int leaf_size = 200;
    double tol = 1e-6;

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
            printf("Usage: %s kid coord_txt l mu tol leaf_size\n", argv[0]);
            printf("kid: Kernel ID, 0 for Gaussian, 1 for Matern 3/2\n");
            printf("l, mu: kernel parameter and diagonal shift\n");
            printf("Optional: leaf_size (default %d), tol (default %e)\n", leaf_size, tol);
        }
        return 255;
    }
    int kid = atoi(argv[1]);
    l  = atof(argv[3]);
    mu = atof(argv[4]);
    if (argc >= 6) leaf_size = atoi(argv[5]);
    if (argc >= 7) tol = atof(argv[6]);
    l2 = l * l;
    _2l2 = l * l * 2.0;
    sqrt3_l = sqrt(3.0) / l;
    if (world.is_root())
    {
        if (kid == 0) printf("Kernel: exp(-|x-y|^2 / (l^2))\n");
        else          printf("Kernel: (1 + k) * exp(-k), k = sqrt(3) / l * |x-y|\n");
        printf("l, mu, tol = %.3f, %.3f, %e\n", l, mu, tol);
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
                return exp(-R2 / l2);
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

        // Create input vector, reference matvec output vector, and structured matrix output vector
        DistributedMatrix<double> x(&grid, npt, 1, rand_matrix_elem);
        DistributedMatrix<double> b_ref(&grid, npt, 1);
        DistributedMatrix<double> b_str(&grid, npt, 1);

        // Vectors for solve test
        DistributedMatrix<double> x2(&grid, npt, 1, rand_matrix_elem);   // StructuredMatrix solve is done in-place
        DistributedMatrix<double> r2(x2);
        double b2_2norm = x2.normF();

        // Set structured matrix options
        options.set_rel_tol(tol);
        options.set_leaf_size(leaf_size);
        options.set_type(structured::Type::HODLR);
        options.set_max_rank(npt);

        // Build a StructuredMatrix
        try {
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

            for (int i = 0; i <= 5; i++)
            {
                st = MPI_Wtime();
                H_->mult(Trans::N, x, b_str);
                et = MPI_Wtime();
                if (i > 0 && my_rank == 0) 
                {
                    printf("StructuredMatrix matvec time = %.3f\n", et - st);
                    fflush(stdout);
                }
            }

            st = MPI_Wtime();
            H_->factor();
            H_->solve(x2);
            et = MPI_Wtime();
            if (my_rank == 0) 
            {
                printf("StructuredMatrix factor and solve time = %.3f\n", et - st);
                fflush(stdout);
            }
        } catch (std::exception& e) {
            if (my_rank == 0)
            {
                cout << "Failed: " << e.what() << endl;
                fflush(stdout);
            }
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

        // Test dense matvec
        for (int i = 0; i <= 5; i++)
        {
            st = MPI_Wtime();
            A2d.mult(Trans::N, x, b_ref);
            et = MPI_Wtime();
            if (i > 0 && my_rank == 0) 
            {
                printf("Dense matvec time = %.3f\n", et - st);
                fflush(stdout);
            }
        }

        st = MPI_Wtime();
        gemv(Trans::N, double(1.0), A2d, x2, double(-1.0), r2);
        et = MPI_Wtime();
        if (my_rank == 0) 
        {
            printf("StructuredMatrix compute solve residual time = %.3f\n", et - st);
            fflush(stdout);
        }

        // Compare matvec and solve error
        double bref_2norm = b_ref.normF();
        DistributedMatrix<double> mv_res = b_ref.scale_and_add(-1.0, b_str);
        double mv_err_2norm = mv_res.normF();
        double solve_err_2norm = r2.normF();
        if (my_rank == 0)
        {
            printf("HOLDR matvec : ||b_ref - b_hss||_2 / ||b_ref||_2 = %e\n", mv_err_2norm / bref_2norm);
            printf("HOLDR solve  : ||A_dense * x - b||_2 / ||b||_2 = %e\n", solve_err_2norm / b2_2norm);
            fflush(stdout);
        }
    }  
    
    scalapack::Cblacs_exit(1);
    MPI_Finalize();
    return 0;
}
