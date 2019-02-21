/**
 * @file    CSRPlus.h
 * @brief   CSRPlus matrix header file 
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2017-2019 Georgia Institute of Technology
 */

#ifndef __CSRPLUS_H__
#define __CSRPLUS_H__

struct CSRPlusMatrix_
{
    // Standard CSR arrays and parameters
    int    nrows, nnz;
    int    *row_ptr;
    int    *col;
    double *val;
    
    // CSRPlus task partitioning information
    int    nblocks;     // Total number of nnz blocks
    int    *nnz_spos;   // First nnz of a block
    int    *nnz_epos;   // Last  nnz (included) of a block
    int    *first_row;  // First row of a block
    int    *last_row;   // Last  row of a block
    int    *fr_intact;  // If the first row of a block is intact
    int    *lr_intact;  // If the last  row of a block is intact
    double *fr_res;     // Partial result of the first row 
    double *lr_res;     // Partial result of the last  row
};

typedef struct CSRPlusMatrix_  CSRPlusMatrix;
typedef struct CSRPlusMatrix_* CSRPlusMatrix_t;

#ifdef __cplusplus
extern "C" {
#endif

// =============== Helper Functions ===============

// Initialize a CSRPlus matrix using a COO matrix
// Note: This function assumes that the input COO matrix is not sorted
// Input:
//   nrows    : Number of rows 
//   nnz      : Number of non-zeros
//   row, col : Indices of non-zero elements
//   val      : Values of non-zero elements
//   CSRP     : Pointer to CSRPlus matrix structure pointer
// Output:
//   CSRP     : Pointer to initialized CSRPlus matrix structure pointer
void CSRP_init_with_COO_matrix(
    const int nrows, const int nnz, const int *row,
    const int *col, const double *val, CSRPlusMatrix_t *CSRP
);

// Free a CSRPlus matrix structure
// Input:
//   CSRP : CSRPlus matrix structure pointer
void CSRP_free(CSRPlusMatrix_t CSRP);

// Partition a CSR matrix into multiple blocks with the same nnz
// Input:
//   nblocks : Number of blocks to be partitioned
//   CSRP    : CSRPlus matrix structure pointer
// Output:
//   CSRP    : CSRPlus matrix structure pointer with partitioning information
void CSRP_partition(const int nblocks, CSRPlusMatrix_t CSRP);

// Use first-touch policy to optimize the storage of CSR arrays in a CSRPlus matrix
// Input:
//   CSRP      : CSRPlus matrix structure pointer
// Output:
//   CSRP      : CSRPlus matrix structure pointer with NUMA optimized storage
void CSRP_optimize_NUMA(CSRPlusMatrix_t CSRP);



// =============== Calculation Kernels ===============

// Perform OpenMP parallelized CSR SpMV with a CSRPlus matrix
// Input:
//   CSRP : CSRPlus matrix structure pointer
//   x    : Input vector
// Output:
//   y    : Output vector
void CSRP_SpMV(CSRPlusMatrix_t CSRP, const double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif
