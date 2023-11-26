#include <iostream>
#include <cmath>
#include <cstdio>
#include <omp.h>
using namespace std;

#include "structured/StructuredMatrix.hpp"
#include "iterative/IterativeSolvers.hpp"
using namespace strumpack;

double *coord = NULL;
int npt, dim;
double l, mu, _2l2;

extern "C" int dgemm_(
    const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
    const double *beta, double *c, const int *ldc
);
extern "C" int dgemv_(
    const char *trans, const int *m, const int *n,
    const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
    const double *beta, double *y, const int *incy
);

void build_denseK_Gaussian(double *denseK)
{
    double *x2 = (double *) malloc(sizeof(double) * npt);
    #pragma omp parallel for
    for (int i = 0; i < npt; i++)
    {
        double tmp = 0;
        double *xi = coord + i * dim;
        #pragma omp simd
        for (int j = 0; j < dim; j++)
            tmp += xi[j] * xi[j];
        x2[i] = tmp;
    }
    printf("Compute X.^2 done\n");
    fflush(stdout);
    double alpha = -2.0, beta = 1.0;
    dgemm_(
        "T", "N", &npt, &npt, &dim,
        &alpha, coord, &dim, coord, &dim,
        &beta, denseK, &npt
    );
    printf("Compute X^T * X done\n");
    fflush(stdout);
    #pragma omp parallel for
    for (int i = 0; i < npt; i++)
    {
        ptrdiff_t offset = (ptrdiff_t) i * (ptrdiff_t) npt;
        double *denseK_i = denseK + offset;
        #pragma omp simd
        for (int j = 0; j < npt; j++)
        {
            double r2 = x2[i] + x2[j] + denseK_i[j];
            denseK_i[j] = exp(-r2 / _2l2);
        }
        denseK_i[i] += mu;
    }
    printf("Compute denseK done\n");
    fflush(stdout);
    free(x2);
}

template<typename scalar_t> void
print_info(const structured::StructuredMatrix<scalar_t>* H,
           const structured::StructuredOptions<scalar_t>& opts)
{
    cout << get_name(opts.type()) << endl;
    cout << "  - nonzeros(H) = " << H->nonzeros() << endl;
    cout << "  - memory(H) = " << H->memory() / 1e6 << " MByte" << endl;
    cout << "  - rank(H) = " << H->rank() << endl;
}

int main(int argc, char *argv[])
{
    int leaf_size = 200;
    double tol = 1e-10;

    if (argc < 4)
    {
        printf("Usage: %s coord_txt l mu tol leaf_size\n", argv[0]);
        printf("Kernel: exp(-|x-y|^2 / (2 * l^2)), kernel matrix: K + mu * I\n");
        printf("Optional: tol (default 1e-10), leaf_size (default 200)\n");
        return 255;
    }
    l  = atof(argv[2]);
    mu = atof(argv[3]);
    if (argc >= 5) tol = atof(argv[4]);
    if (argc >= 6) leaf_size = atoi(argv[5]);
    _2l2 = l * l * 2.0;

    FILE *inf = fopen(argv[1], "r");
    fscanf(inf, "%d %d", &npt, &dim);

    printf("npt, dim   = %d, %d\n", npt, dim);
    printf("l, mu, tol = %.3f, %.3f, %e\n", l, mu, tol);
    fflush(stdout);

    coord = (double *) malloc(sizeof(double) * npt * dim);
    for (int i = 0; i < npt * dim; i++) fscanf(inf, "%lf", coord + i);
    fclose(inf);
    printf("\nRead point coordinates from input file done\n");
    fflush(stdout);

    structured::StructuredOptions<double> options;
    //options.set_verbose(false);

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
    auto GaussianBlock =
        [&MyGaussianKernel](const std::vector<std::size_t>& I,
                            const std::vector<std::size_t>& J,
                            DenseMatrix<double>& B)
    {
        for (int j = 0; j < J.size(); j++)
            for (int i = 0; i < I.size(); i++)
                B(i, j) = MyGaussianKernel(I[i], J[j]);
    };

    // Generate a random input vector
    DenseMatrix<double> x(npt, 1), b_ref(npt, 1), b_hss(npt, 1);
    double *x_data = x.data();
    double *b_ref_data = b_ref.data();
    double *b_hss_data = b_hss.data();
    for (int i = 0; i < npt; i++)
        x_data[i] = 2.0 * ((double) rand() / (double) RAND_MAX) - 1.0;
    printf("Allocate vectors done\n");
    fflush(stdout);

    DenseMatrix<double> A(npt, npt);
    printf("Allocate denseK done\n");
    fflush(stdout);
    auto Amult = [&A](Trans t, const DenseMatrix<double>& R, DenseMatrix<double>& S)
    {
        gemm(t, Trans::N, double(1.), A, R, double(0.), S);
    };

    // Construct a dense matrix A and compute b_ref := A * x
    double st, et;
    {
        st = omp_get_wtime();
        build_denseK_Gaussian(A.data());
        et = omp_get_wtime();
        printf("Dense kernel matrix construction used %.3f s\n", et - st);
        fflush(stdout);

        st = omp_get_wtime();
        //gemv(Trans::N, 1.0, A, x, 0.0, b_ref);
        double alpha = 1.0, beta = 0.0;
        int ione = 1;
        dgemv_("N", &npt, &npt, &alpha, A.data(), &npt, x_data, &ione, &beta, b_ref_data, &ione);
        et = omp_get_wtime();
        printf("Dense kernel matrix-vector multiplication used %.3f s\n", et - st);
        fflush(stdout);
    }

    // Construct a HSS matrix H and compute b_hss := H * x
    {
        options.set_rel_tol(tol);
        options.set_leaf_size(leaf_size);
        options.set_type(structured::Type::HSS);
        options.set_max_rank(npt);

        st = omp_get_wtime();
        // construct_from_elements may need A LOT OF MEMORY 
        //auto H = structured::construct_from_elements<double>(npt, npt, GaussianBlock, options);
        auto H = structured::construct_partially_matrix_free<double>(npt, npt, Amult, GaussianBlock, options);
        et = omp_get_wtime();
        printf("HSS matrix construction used %.3f s\n", et - st);
        print_info(H.get(), options);
        fflush(stdout);

        // HSS matvec
        const structured::StructuredMatrix<double>* H_ = H.get();
        st = omp_get_wtime();
        H_->mult(Trans::N, x, b_hss);
        et = omp_get_wtime();
        printf("HSS matrix-vector multiplication used %.3f s\n", et - st);
        fflush(stdout);
    }

    // Compute relative error
    double ref_2norm = 0, err_2norm = 0;
    for (int i = 0; i < npt; i++)
    {
        double err = b_ref_data[i] - b_hss_data[i];
        ref_2norm += b_ref_data[i] * b_ref_data[i];
        err_2norm += err * err;
    }
    ref_2norm = sqrt(ref_2norm);
    err_2norm = sqrt(err_2norm);
    printf("||b_ref - b_hss||_2 / ||b_ref||_2 = %e\n", err_2norm / ref_2norm);

    free(coord);
    return 0;
}
