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

void GMRES(
    const Mat &A, HODLR *M, const Mat &b, const int n, 
    const double GMRES_tol, const int max_iter, const int m
)
{
    Mat V = (Mat::Zero(n, m+1)).real();
    Mat H = (Mat::Zero(m+1, m)).real();
    Mat Z = (Mat::Zero(n, m+1)).real();
    Mat v = (Mat::Zero(n, 1)).real();
    Mat x = (Mat::Zero(n, 1)).real();
    Mat z = (Mat::Zero(n, 1)).real();
    Mat c  = (Mat::Zero(m+1, 1)).real();
    Mat s  = (Mat::Zero(m+1, 1)).real();
    Mat rs = (Mat::Zero(m+1, 1)).real();

    double st = omp_get_wtime();

    double b_2norm = b.norm();
    double stop_2norm = b_2norm * GMRES_tol;
    printf("||b||_2 = %e\n", b_2norm);
    printf("Stopping ||r||_2 = %e\n", stop_2norm);
    printf("Iter    relres\n");

    char *noprec = getenv("NOPREC");
    int noprec_flag = (noprec != NULL) ? atoi(noprec) : 0;
    if (noprec_flag) printf("Will ignore preconditioner\n");

    int iter = 0;
    while (iter < max_iter)
    {
        v = b - A * x;
        for (int l = 0; l < n; l++) V(l, 0) = v(l);

        double ro = v.norm();
        if ((ro <= stop_2norm) || (iter >= max_iter))
        {
            printf("%4d    %e\n", iter, ro / b_2norm);
            printf("GMRES converged\n");
            fflush(stdout);
            break;
        }

        if (iter > 0) stop_2norm = GMRES_tol;  // w/o this line gmres was stopping too soon
        double t = 1.0 / ro;
        for (int l = 0; l < n; l++) V(l, 0) = V(l, 0) * t;
        rs(0) = ro;
        printf("%4d    %e\n", iter, ro / b_2norm);
        fflush(stdout);

        int i = 0;
        while ((i < m) && (ro > stop_2norm) && (iter < max_iter))
        {
            i++;
            iter++;
            int i1 = i + 1;

            for (int l = 0; l < n; l++) v(l) = V(l, i-1);
            if (noprec_flag) z = v;
            else z = M->solve(v);
            v = A * z;
            for (int l = 0; l < n; l++)
            {
                Z(l, i-1) = z(l);
                V(l, i1-1) = v(l);
            }

            for (int j = 1; j <= i; j++)
            {
                double t = 0;
                for (int l = 0; l < n; l++) t = t + V(l, j-1) * V(l, i1-1);
                H(j-1, i-1) = t;
                for (int l = 0; l < n; l++) V(l, i1-1) = V(l, i1-1) - t * V(l, j-1);
            }

            for (int l = 0; l < n; l++) v(l) = V(l, i1-1);
            double t = v.norm();
            H(i1-1, i-1) = t;
            if (t > 1e-15)
            {
                t = 1.0 / t;
                for (int l = 0; l < n; l++) V(l, i1-1) *= t;
            }

            if (i > 1)
            {
                for (int k = 2; k <= i; k++)
                {
                    int k1 = k - 1;
                    t = H(k1-1, i-1);
                    H(k1-1, i-1) =  c(k1-1) * t + s(k1-1) * H(k-1, i-1);
                    H(k-1,  i-1) = -s(k1-1) * t + c(k1-1) * H(k-1, i-1);
                }
            }

            double gamma = H(i-1, i-1) * H(i-1, i-1) + H(i1-1, i-1) * H(i1-1, i-1);
            gamma = sqrt(gamma);
            if (gamma < 1e-15) gamma = 1e-15;
            c(i-1) = H(i-1, i-1)  / gamma;
            s(i-1) = H(i1-1, i-1) / gamma;
            rs(i1-1) = -s(i-1) * rs(i-1);
            rs(i-1)  =  c(i-1) * rs(i-1);

            H(i-1, i-1) = c(i-1) * H(i-1, i-1) + s(i-1) * H(i1-1, i-1);
            ro = abs(rs(i1-1));
            printf("%4d    %e\n", iter, ro / b_2norm);
            fflush(stdout);
        }  // End of "while ((ii < m) && (ro > stop_2norm) && (iter < max_iter))"

        rs(i-1) = rs(i-1) / H(i-1, i-1);
        for (int k = i-1; k >= 1; k--)
        {
            double t = rs(k-1);
            for (int j = k+1; j <= i; j++) t -= H(k-1, j-1) * rs(j-1);
            rs(k-1) = t / H(k-1, k-1);
        }

        for (int j = 1; j <= i; j++)
            for (int l = 0; l < n; l++) x(l, 0) += rs(j-1) * Z(l, j-1);
    }  // End of "while (iter < max_iter)"

    double et = omp_get_wtime();
    printf("GMRES: performed %d iterations, time = %.2f s\n", iter, et - st);
    v = b - A * x;
    printf("Final relative residual = %e\n", v.norm() / b_2norm);
}

int main(int argc, char* argv[])
{
    int npt, dim, leaf_size = 200, max_iter = 500, restart = 25;
    double l, mu, HODLR_tol = 1e-4, GMRES_tol = 1e-4;

    if (argc < 4)
    {
        printf("Usage: %s coord_txt l mu leaf_size HODLR_tol GMRES_tol max_iter restart\n", argv[0]);
        printf("Kernel: exp(-|x-y|^2 / (l^2)), kernel matrix: K + mu * I\n");
        printf("Optional: leaf_size (default %d), HODLR_tol (default %e)\n", leaf_size, HODLR_tol);
        printf("          GMRES_tol (default %e), max_iter (default %d), restart (default %d)\n", GMRES_tol, max_iter, restart);
        return 255;
    }
    l  = atof(argv[2]);
    mu = atof(argv[3]);
    if (argc >= 5) leaf_size = atoi(argv[4]);
    if (argc >= 6) HODLR_tol = atof(argv[5]);
    if (argc >= 7) GMRES_tol = atof(argv[6]);
    if (argc >= 8) max_iter  = atoi(argv[7]);
    if (argc >= 9) restart   = atoi(argv[8]);

    FILE *inf = fopen(argv[1], "r");
    fscanf(inf, "%d %d", &npt, &dim);

    std::cout << "========================= Problem Parameters =========================" << std::endl;
    std::cout << "Matrix Size                        : " << npt << std::endl;
    std::cout << "Dimensionality                     : " << dim << std::endl;
    std::cout << "Kernel parameters l, mu            : " << l << ", " << mu << std::endl;
    std::cout << "Leaf Size                          : " << leaf_size << std::endl;
    std::cout << "HODLR compression tolerance        : " << HODLR_tol << std::endl;
    std::cout << "GMRES reltol                       : " << GMRES_tol << std::endl;
    std::cout << "GMRES max iter                     : " << max_iter << std::endl;
    std::cout << "GMRES restart iter                 : " << restart << std::endl << std::endl;

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

    //srand(2024);
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

    std::cout << "========================= Precond GMRES =========================" << std::endl;
    Mat b = (Mat::Random(npt, 1)).real();
    GMRES(B, T, b, npt, GMRES_tol, max_iter, restart);

    free(coord);
    delete T;
    delete K;

    return 0;
}