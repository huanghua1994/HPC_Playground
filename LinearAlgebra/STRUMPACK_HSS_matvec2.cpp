
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kernel/KernelRegression.hpp"
#include "misc/TaskTimer.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;
using namespace strumpack::kernel;


template<typename scalar_t> vector<scalar_t>
read_from_file(string filename) {
  vector<scalar_t> data;
  ifstream f(filename);
  string l;
  while (getline(f, l)) {
    istringstream sl(l);
    string s;
    while (getline(sl, s, ','))
      data.push_back(stod(s));
  }
  data.shrink_to_fit();
  return data;
}


int main(int argc, char *argv[]) {
  using scalar_t = double;
  string filename("./data/susy_10Kn_train.csv");
  size_t d = 8;
  scalar_t h = 1.3;
  scalar_t lambda = 3.11;
  scalar_t reltol = 1e-2;
  int p = 1;  // kernel degree
  KernelType ktype = KernelType::GAUSS;

  cout << "# usage: ./HSS file d h lambda reltol"  << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) lambda = stof(argv[4]);
  if (argc > 5) reltol = stof(argv[5]);

  cout << endl;
  cout << "# file            = " << filename << endl;
  cout << "# data dimension  = " << d << endl;
  cout << "# kernel h        = " << h << endl;
  cout << "# lambda          = " << lambda << endl;
  cout << "# HSS reltol      = " << reltol << endl;

  TaskTimer timer("compression");

  // Read from csv file
  vector<scalar_t> coord = read_from_file<scalar_t>(filename);
  size_t n = coord.size() / d;
  vector<scalar_t> coord0(n);
  for (int i = 0; i < n; i++) coord0[i] = coord[i];

  // Set up a kernel 
  DenseMatrixWrapper<scalar_t> Xcoord(d, n, coord.data(), d);
  auto krnl = create_kernel<scalar_t>(ktype, Xcoord, h, lambda, p);

  // Create a dense kernel matrix (row-major)
  vector<scalar_t> dense_K(n * n);
  timer.start();
  #pragma omp parallel for
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      dense_K[i * n + j] = krnl->eval(i, j);
  cout << "# build dense kernel matrix took " << timer.elapsed() << endl;

  // Create a random input vector u
  strumpack::DenseMatrix<scalar_t> u(n, 1);
  srand(19241112);
  for (int i = 0; i < n; i++)
    u(i, 0) = ((scalar_t) rand() / (scalar_t) RAND_MAX) * 2.0 - 1.0;

  // Compute v_ref = dense_K * u
  vector<scalar_t> v_ref(n);
  #pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    v_ref[i] = 0;
    for (int j = 0; j < n; j++)
      v_ref[i] += dense_K[i * n + j] * u(j, 0);
  }

  // Build a HSS matrix 
  HSSOptions<scalar_t> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);
  hss_opts.set_rel_tol(reltol);
  hss_opts.describe_options();
  cout << "Constructing HSS matrix... " << std::endl;
  timer.start();
  HSS::HSSMatrix<scalar_t> H(*krnl, hss_opts);
  cout << "# build HSS matrix took " << timer.elapsed() << endl;
  if (H.is_compressed())
  {
    std::cout << "# created HSS matrix of dimension "
              << H.rows() << " x " << H.cols()
              << " with " << H.levels() << " levels" << std::endl
              << "# compression succeeded!" << std::endl
              << "# rank(H) = " << H.rank() << std::endl
              << "# HSS memory(H) = "
              << H.memory() / 1e6 << " MB " << std::endl << std::endl;
  }

  // Test HSS matvec accuracy
  strumpack::DenseMatrix<scalar_t> v_hss(n, 1);
  v_hss = H.apply(u);
  scalar_t vref_2norm = 0, err_2norm = 0;
  for (int i = 0; i < n; i++)
  {
    scalar_t diff = v_ref[i] - v_hss(i, 0);
    vref_2norm += v_ref[i] * v_ref[i];
    err_2norm += diff * diff;
  }
  vref_2norm = std::sqrt(vref_2norm);
  err_2norm = std::sqrt(err_2norm);
  printf("HSSMatrix matvec relerr = %e\n", err_2norm / vref_2norm);

  // Write vectors to files
  FILE *ouf = NULL;
  ouf = fopen("u.txt", "w");
  for (int i = 0; i < n; i++)
    fprintf(ouf, "%.12f\n", u(i, 0));
  fclose(ouf);

  ouf = fopen("v_ref.txt", "w");
  for (int i = 0; i < n; i++)
    fprintf(ouf, "%.12f\n", v_ref[i]);
  fclose(ouf);

  ouf = fopen("v.txt", "w");
  for (int i = 0; i < n; i++)
    fprintf(ouf, "%.12f\n", v_hss(i, 0));
  fclose(ouf);

  timer.start();
  H.factor();
  std::cout << "# factorization time = " << timer.elapsed() << std::endl;
  
  strumpack::DenseMatrix<scalar_t> u_solve(n, 1);
  for (int i = 0; i < n; i++) u_solve(i, 0) = v_ref[i];
  timer.start();
  H.solve(u_solve);
  std::cout << "# solve time = " << timer.elapsed() << std::endl;
  
  vector<scalar_t> v_solve(n);
  #pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    v_solve[i] = 0;
    for (int j = 0; j < n; j++)
      v_solve[i] += dense_K[i * n + j] * u_solve(j, 0);
  }
  err_2norm = 0;
  for (int i = 0; i < n; i++)
  {
    scalar_t diff = v_solve[i] - v_ref[i];
    err_2norm += diff * diff;
  }
  err_2norm = std::sqrt(err_2norm);
  printf("HSS solve ||A * x - b||_2 / ||b||_2 = %e\n", err_2norm / vref_2norm);

  return 0;
}