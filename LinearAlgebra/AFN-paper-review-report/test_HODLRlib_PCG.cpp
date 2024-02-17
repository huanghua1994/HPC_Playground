// Modified from https://github.com/SAFRAN-LAB/HODLR/blob/master/examples/tutorial.cpp

#include "HODLR_Matrix.hpp"
#include "HODLR.hpp"
#include "KDTree.hpp"

class Kernel : public HODLR_Matrix
{
private:
    Mat x, tx;
    int npt, dim;
    dtype l2, _2l2, mu;


public:
    double kd_time;

    Kernel(int _npt, int _dim, double *coord, double _l, double _mu) : HODLR_Matrix(npt)
    {
        npt  = _npt;
        dim  = _dim;
        l2   = _l * _l;
        _2l2 = _l * _l * 2.0;
        mu   = _mu;

        // getKDTreeSorted requires the coordinate of each point to be stored in a row
        x = (Mat::Zero(npt, dim)).real();
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < npt; j++)
                x(j, i) = coord[j * dim + i];

        double start = omp_get_wtime();
        getKDTreeSorted(x, 0);
        double end = omp_get_wtime();
        kd_time = end - start;
        printf("Build KDTree for %d * %d input points done, time = %.2f s\n", npt, dim, kd_time);

        // Store a transposed x for faster getMatrixEntry
        tx = (Mat::Zero(dim, npt)).real();
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < npt; j++)
                tx(i, j) = x(j, i);
    };

    dtype getMatrixEntry(int i, int j)
    {
        if (i == j) return (1 + mu);
        else
        {
            dtype R2 = 0;
            for (int k = 0; k < dim; k++)
            {
                dtype diff = tx(k, i) - tx(k, j);
                R2 += diff * diff;
            }
            return exp(-R2 / l2);
        }
    }

    ~Kernel() {};
};

void PCG(
    const Mat &A, HODLR *M, const Mat &b,
    const int n, const double PCG_tol, const int max_iter
)
{
    Mat x = (Mat::Zero(n, 1)).real();
    Mat r = (Mat::Zero(n, 1)).real();
    Mat z = (Mat::Zero(n, 1)).real();
    Mat p = (Mat::Zero(n, 1)).real();
    Mat s = (Mat::Zero(n, 1)).real();

    double st = omp_get_wtime();

    // r = b - A * x
    r = b - A * x;

    double r_2norm = r.norm();
    double b_2norm = b.norm();
    double stop_2norm = b_2norm * PCG_tol;
    printf(
        "PCG:\nmax_iter = %d\n||b||_2 = %e\ninitial  ||r||_2 = %e\nstopping ||r||_2 = %e\n", 
        max_iter, b_2norm, r_2norm, stop_2norm
    );
    printf("Iter    relres\n");

    int iter = 0;
    double alpha, beta, rho0, tmp, rho = 1.0;
    while (iter < max_iter && r_2norm > stop_2norm)
    {
        // z = M \ r;
        z = M->solve(r);

        // rho0 = rho;
        // rho  = r' * z;
        // beta = rho / rho0;
        rho0 = rho;
        rho = 0;
        for (int i = 0; i < n; i++) rho += r(i, 0) * z(i, 0);
        beta = rho / rho0;

        // p = z + beta * p; or p = z;
        if (iter == 0)
        {
            for (int i = 0; i < n; i++) p(i, 0) = z(i, 0);
        } else {
            for (int i = 0; i < n; i++) p(i, 0) = z(i, 0) + beta * p(i, 0);
        }

        // s = A * p;
        s = A * p;

        // alpha = rho / (p' * s);
        tmp = 0;
        for (int i = 0; i < n; i++) tmp += p(i, 0) * s(i, 0);
        alpha = rho / tmp;

        // x = x + alpha * p;
        // r = r - alpha * s;
        for (int i = 0; i < n; i++)
        {
            x(i, 0) = x(i, 0) + alpha * p(i, 0);
            r(i, 0) = r(i, 0) - alpha * p(i, 0);
        }
        r_2norm = r.norm();

        iter++;
        printf("%4d    %e\n", iter, r_2norm / b_2norm);
    }

    double et = omp_get_wtime();
    printf("Performed %d PCG iterations, time = %.2f s\n", iter, et - st);
}

int main(int argc, char* argv[])
{
    int npt, dim, leaf_size = 200, max_iter = 500;
    double l, mu, HODLR_tol = 1e-4, PCG_tol = 1e-4;

    if (argc < 4)
    {
        printf("Usage: %s coord_txt l mu leaf_size HODLR_tol PCG_tol max_iter\n", argv[0]);
        printf("Kernel: exp(-|x-y|^2 / (l^2)), kernel matrix: K + mu * I\n");
        printf("Optional: leaf_size (default %d), HODLR_tol (default %e)\n", leaf_size, HODLR_tol);
        printf("          PCG_tol (default %e), max_iter (default %d)\n", PCG_tol, max_iter);
        return 255;
    }
    l  = atof(argv[2]);
    mu = atof(argv[3]);
    if (argc >= 5) leaf_size = atoi(argv[4]);
    if (argc >= 6) HODLR_tol = atof(argv[5]);
    if (argc >= 7) PCG_tol   = atof(argv[6]);
    if (argc >= 8) max_iter  = atoi(argv[7]);

    FILE *inf = fopen(argv[1], "r");
    fscanf(inf, "%d %d", &npt, &dim);

    std::cout << "========================= Problem Parameters =========================" << std::endl;
    std::cout << "Matrix Size                        : " << npt << std::endl;
    std::cout << "Dimensionality                     : " << dim << std::endl;
    std::cout << "Kernel parameters l, mu            : " << l << ", " << mu << std::endl;
    std::cout << "Leaf Size                          : " << leaf_size << std::endl;
    std::cout << "HODLR compression tolerance        : " << HODLR_tol << std::endl << std::endl;
    std::cout << "PCG reltol                         : " << PCG_tol << std::endl;
    std::cout << "PCG max iter                       : " << max_iter << std::endl << std::endl;

    // coord are in col-major, each column is a point coordinate
    double *coord = (double *) malloc(sizeof(double) * npt * dim);
    for (int i = 0; i < npt * dim; i++) fscanf(inf, "%lf", coord + i);
    fclose(inf);
    printf("\nRead point coordinates from input file done\n");
    Kernel *K = new Kernel(npt, dim, coord, l, mu);

    double start, end;

    double hodlr_time, exact_time;
    std::cout << "========================= Assembly Time =========================" << std::endl;
    start = omp_get_wtime();
    // Is it multithreaded?
    Mat B = K->getMatrix(0, 0, npt, npt);
    end   = omp_get_wtime();
    exact_time = (end - start);
    std::cout << "Time for direct matrix generation  : " << exact_time << std::endl;

    bool is_sym = true;
    bool is_pd  = false;  // true?
    start = omp_get_wtime();
    HODLR *T = new HODLR(npt, leaf_size, HODLR_tol);
    T->assemble(K, "rookPivoting", is_sym, is_pd);
    end = omp_get_wtime();
    hodlr_time = (end - start);
    std::cout << "Time for assembly in HODLR form    : " << hodlr_time + K->kd_time << std::endl;
    std::cout << "Magnitude of Speed-Up              : " << (exact_time / hodlr_time) << std::endl << std::endl;

    Mat x = (Mat::Random(npt, 1)).real();
    Mat y_fast, b_fast;

    std::cout << "========================= Matrix-Vector Multiplication =========================" << std::endl;
    start = omp_get_wtime();
    Mat b_exact = B * x;
    end   = omp_get_wtime();
    exact_time = (end - start);
    std::cout << "Time for direct MatVec             : " << exact_time << std::endl;

    start  = omp_get_wtime();
    b_fast = T->matmatProduct(x);
    end    = omp_get_wtime();
    hodlr_time = (end - start);
    std::cout << "Time for MatVec in HODLR form      : " << hodlr_time << std::endl;
    std::cout << "Magnitude of Speed-Up              : " << (exact_time / hodlr_time) << std::endl;
    std::cout << "MatVec relative error              : " << (b_fast-b_exact).norm() / (b_exact.norm()) << std::endl << std::endl;

    std::cout << "========================= Solving =========================" << std::endl;
    Mat x_fast;
    start  = omp_get_wtime();
    T->factorize();
    x_fast = T->solve(b_exact);
    end    = omp_get_wtime();
    hodlr_time = (end - start);
    std::cout << "Time to solve HODLR form           :" << hodlr_time << std::endl;
    std::cout << "HODLR backward error               :" << (T->matmatProduct(x_fast) - b_exact).norm() / b_exact.norm() << std::endl;
    std::cout << "Solve relative error               :" << (B * x_fast - b_exact).norm() / b_exact.norm() << std::endl;

    std::cout << "========================= Precond CG =========================" << std::endl;
    Mat b = (Mat::Random(npt, 1)).real();
    PCG(B, T, b, npt, PCG_tol, max_iter);

    free(coord);
    delete T;
    delete K;

    return 0;
}