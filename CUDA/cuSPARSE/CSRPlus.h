/**
 * @file    CSRPlus.h
 * @brief   CSRPlus format 
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2017-2019 Georgia Institute of Technology
 */

#ifndef __CSRPLUS_H__
#define __CSRPLUS_H__

struct CSRP_blk_info_
{
	int    nnz_spos,  nnz_epos;   // First and last nnz of this block (INCLUDED the last nnz)
    int    first_row, last_row;   // First and last row of this block
	int    fr_intact, lr_intact;  // If the first / last row of this block is intact
	double fr_res,    lr_res;     // Partial result for the first / last row
};

typedef struct CSRP_blk_info_  CSRP_blk_info;
typedef struct CSRP_blk_info_* CSRP_blk_info_t;

extern "C" {

// Partition a CSR matrix into multiple blocks with the same nnz
// Input:
//   nnz       : Number of non-zeros of the matrix
//   nblocks   : Number of blocks to be partitioned
//   nrows     : Number of rows of the matrix
//   row_ptr   : Displacement of each row's nnz, length nrows+1, standard array in CSR
// Output:
//   blk_info  : Partition information of each block, length nblocks, should be allocated before 
void CSRP_equal_nnz_partition(
    const int nnz, const int nblocks, const int nrows,
    const int *row_ptr, CSRP_blk_info_t blk_info
);

// Perform OpenMP paralleled CSR SpMV using partitioned block information
// Input:
//   row_ptr   : Displacement of each row's non-zeros, length nrows+1, standard array in CSR
//   col_idx   : Column indices of non-zeros, standard array in CSR
//   val       : Non-zero values, standard array in CSR
//   nrows     : Number of rows of the sparse matrix
//   blk_info  : Partition information of each block, length nblocks, should be allocated before 
//   nblocks   : Number of blocks to be partitioned
//   x         : Input vector
// Output:
//   y         : Output vector
void CSRP_SpMV(
    const int *row_ptr, const int *col_idx, const double *val, 
    const int nrows, const int nblocks, CSRP_blk_info_t blk_info, 
    const double *x, double *y
);

}

#endif
