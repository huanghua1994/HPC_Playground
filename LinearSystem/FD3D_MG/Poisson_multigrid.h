#ifndef __POISSON_MULTIGRID_H__
#define __POISSON_MULTIGRID_H__

#include "CSR_mat.h"

struct mg_data_
{
    int    nlevel;
    int    *vec_len;
    int    *lastA_ipiv;
    double *lastA_LU;
    double **ev;
    double **rv;
    double **tv;
    double **M;
    CSR_mat_t *A;
    CSR_mat_t *R;
    CSR_mat_t *P;
};

typedef struct mg_data_* mg_data_t;

#ifdef __cplusplus
extern "C" {
#endif

void MG_init(
    const double *cell_dims, const int *grid_sizes, const int *BCs, 
    const int FDn, mg_data_t *mg_data_
);

void MG_destroy(mg_data_t mg_data);

void MG_solve(mg_data_t mg_data, const double *b, double *x, const double reltol);

#ifdef __cplusplus
};
#endif

#endif

