/**
 * @file    CSRPlus_kernels.c
 * @brief   CSRPlus matrix SpMV / SpMM kernels
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2017-2019 Georgia Institute of Technology
 */

#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#include <omp.h>

#include "CSRPlus.h"

// =============== SpMV kernels ===============

static double CSR_SpMV_row_seg(
    const int seg_len, const int *__restrict col, 
    const double *__restrict val, const double *__restrict x
)
{
    double res = 0.0;
    #pragma omp simd
    for (int idx = 0; idx < seg_len; idx++)
        res += val[idx] * x[col[idx]];
    return res;
}

static void CSR_SpMV_row_block(
    const int srow, const int erow,
    const int *row_ptr, const int *col, const double *val, 
    const double *__restrict x, double *__restrict y
)
{
    for (int irow = srow; irow < erow; irow++)
    {
        double res = 0.0;
        #pragma omp simd
        for (int idx = row_ptr[irow]; idx < row_ptr[irow + 1]; idx++)
            res += val[idx] * x[col[idx]];
        y[irow] = res;
    }
}

static void CSRP_SpMV_block(CSRPlusMatrix_t CSRP, const int iblock, const double *x, double *y)
{
    int    *row_ptr = CSRP->row_ptr;
    int    *col     = CSRP->col;
    double *val     = CSRP->val;
    
    if (CSRP->first_row[iblock] == CSRP->last_row[iblock])
    {
        // This thread handles a segment on 1 row
        
        int nnz_spos = CSRP->nnz_spos[iblock];
        int nnz_epos = CSRP->nnz_epos[iblock];
        int seg_len  = nnz_epos - nnz_spos + 1;
        
        CSRP->fr_res[iblock] = CSR_SpMV_row_seg(seg_len, col + nnz_spos, val + nnz_spos, x);
        CSRP->lr_res[iblock] = 0.0;
    } else {
        // This thread handles segments on multiple rows
        
        int first_intact_row = CSRP->first_row[iblock];
        int last_intact_row  = CSRP->last_row[iblock];
        
        if (CSRP->fr_intact[iblock] == 0)
        {
            int nnz_spos = CSRP->nnz_spos[iblock];
            int nnz_epos = row_ptr[first_intact_row + 1];
            int seg_len  = nnz_epos - nnz_spos;
            
            CSRP->fr_res[iblock] = CSR_SpMV_row_seg(seg_len, col + nnz_spos, val + nnz_spos, x);
            first_intact_row++;
        }
        
        if (CSRP->lr_intact[iblock] == 0)
        {
            int nnz_spos = row_ptr[last_intact_row];
            int nnz_epos = CSRP->nnz_epos[iblock];
            int seg_len  = nnz_epos - nnz_spos + 1;
            
            CSRP->lr_res[iblock] = CSR_SpMV_row_seg(seg_len, col + nnz_spos, val + nnz_spos, x);
            last_intact_row--;
        }
        
        CSR_SpMV_row_block(
            first_intact_row, last_intact_row + 1,
            row_ptr, col, val, x, y
        );
    }
}

// Perform OpenMP parallelized CSR SpMV with a CSRPlus matrix
void CSRP_SpMV(CSRPlusMatrix_t CSRP, const double *x, double *y)
{
    int nrows = CSRP->nrows;
    int nblocks = CSRP->nblocks;
    
    #pragma omp parallel 
    {
        #pragma omp for 
        for (int i = 0; i < nrows; i++) y[i] = 0.0;
        
        #pragma omp for
        for (int iblock = 0; iblock < nblocks; iblock++)
            CSRP_SpMV_block(CSRP, iblock, x, y);
    }
    
    // Accumulate the results for the threads that shared the same row
    for (int iblock = 0; iblock < nblocks; iblock++)
    {
        if (CSRP->fr_intact[iblock] == 0)
        {
            int first_row = CSRP->first_row[iblock];
            y[first_row] += CSRP->fr_res[iblock];
        }
        
        if (CSRP->lr_intact[iblock] == 0)
        {
            int last_row = CSRP->last_row[iblock];
            y[last_row] += CSRP->lr_res[iblock];
        }
    }
}
