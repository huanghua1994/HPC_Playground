// Modified from https://github.com/SAFRAN-LAB/HODLR/blob/master/examples/tutorial.cpp

#include "HODLR_Matrix.hpp"
#include "HODLR.hpp"
#include "KDTree.hpp"

class Kernel : public HODLR_Matrix
{

private:
    Mat x;
    int npt, dim;
    dtype _2l2, mu;


public:
    double kd_time;

    Kernel(int _npt, int _dim, double *coord, double _l, double _mu) : HODLR_Matrix(npt)
    {
        npt  = _npt;
        dim  = _dim;
        _2l2 = _l * _l * 2.0;
        mu   = _mu;

        // Eigen uses col-major; store each point contiguously 
        x = (Mat::Zero(dim, npt)).real();
        for (int j = 0; j < npt; j++)
            for (int i = 0; i < dim; i++)
                x(i, j) = coord[j * dim + i];

        double start = omp_get_wtime();
        getKDTreeSorted(x, 0);
        double end = omp_get_wtime();
        kd_time = end - start;
        printf("Build KDTree for %d * %d input points done, time = %.2f s\n", npt, dim, kd_time);
    };

    dtype getMatrixEntry(int i, int j)
    {
        if (i == j) return (1 + mu);
        else
        {
            dtype R2 = 0;
            for (int k = 0; k < dim; k++)
            {
                dtype diff = x(k, i) - x(k, j);
                R2 += diff * diff;
            }
            return exp(-R2 / _2l2);
        }
    }

    ~Kernel() {};
};

int main(int argc, char* argv[])
{
    int npt, dim, leaf_size = 200;
    double l, mu, tol = 1e-10;

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

    FILE *inf = fopen(argv[1], "r");
    fscanf(inf, "%d %d", &npt, &dim);

    std::cout << "========================= Problem Parameters =========================" << std::endl;
    std::cout << "Matrix Size                        : " << npt << std::endl;
    std::cout << "Dimensionality                     : " << dim << std::endl;
    std::cout << "l, mu                              : " << l << ", " <<mu << std::endl;
    std::cout << "Leaf Size                          : " << leaf_size << std::endl;
    std::cout << "Tolerance                          : " << tol << std::endl << std::endl;

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
    bool is_pd  = true;
    start = omp_get_wtime();
    HODLR *T = new HODLR(npt, leaf_size, tol);
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
    std::cout << "Error in the solution is           : " << (b_fast-b_exact).norm() / (b_exact.norm()) << std::endl << std::endl;

    free(coord);
    delete T;
    delete K;

    return 0;
}