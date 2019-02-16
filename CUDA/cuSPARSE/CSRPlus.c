/**
 * @file    CSRPlus.h
 * @brief   CSRPlus format 
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2017-2019 Georgia Institute of Technology
 */

#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <omp.h>

#include "CSRPlus.h"

static void partitionBlocks(const int nelem, const int nblocks, int *displs)
{
    int bs0 = nelem / nblocks;
    int bs1 = bs0;
    int remainder = nelem % nblocks;
    if (remainder > 0) bs1++;
    displs[0] = 0;
    for (int i = 0; i < remainder; i++)
        displs[i + 1] = displs[i] + bs1;
    for (int i = remainder; i < nblocks; i++)
        displs[i + 1] = displs[i] + bs0;
}

static int lower_bound(const int *a, int n, int x) 
{
    int l = 0, h = n;
    while (l < h) 
    {
        int mid = l + (h - l) / 2;
        if (x <= a[mid]) h = mid;
        else l = mid + 1;
    }
    return l;
}

// Partition a CSR matrix into multiple blocks with the same nnz
void CSRP_equal_nnz_partition(
    const int nnz, const int nblocks, const int nrows,
    const int *row_ptr, CSRP_blk_info_t blk_info
)
{
	int *nnz_displs = (int *) malloc((nblocks + 1) * sizeof(int));
    partitionBlocks(nnz, nblocks, nnz_displs);
    
	for (int iblock = 0; iblock < nblocks; iblock++)
	{
		int block_nnz_spos = nnz_displs[iblock];
		int block_nnz_epos = nnz_displs[iblock + 1] - 1;
		int spos_in_row = lower_bound(row_ptr, nrows + 1, block_nnz_spos);
		int epos_in_row = lower_bound(row_ptr, nrows + 1, block_nnz_epos);
		if (row_ptr[spos_in_row] > block_nnz_spos) spos_in_row--;
		if (row_ptr[epos_in_row] > block_nnz_epos) epos_in_row--;
		
		// Note: It is possible that the last nnz is the first nnz in a row,
		// and there are some empty rows between the last row and previous non-empty row
		while (row_ptr[epos_in_row] == row_ptr[epos_in_row + 1]) epos_in_row++;  
		
		blk_info[iblock].nnz_spos  = block_nnz_spos;
		blk_info[iblock].nnz_epos  = block_nnz_epos;
		blk_info[iblock].first_row = spos_in_row;
		blk_info[iblock].last_row  = epos_in_row;
		
		if ((epos_in_row - spos_in_row) >= 1)
		{
            int fr_intact = (block_nnz_spos == row_ptr[spos_in_row]);
            int lr_intact = (block_nnz_epos == row_ptr[epos_in_row + 1] - 1);
            
            blk_info[iblock].fr_intact = fr_intact;
            blk_info[iblock].lr_intact = lr_intact;
		} else {
            // Mark that this thread only handles a segment of a row
			blk_info[iblock].fr_intact =  0;
			blk_info[iblock].lr_intact = -1;  
		}
	}
	
	blk_info[nblocks - 1].last_row = nrows - 1;
	blk_info[nblocks - 1].nnz_epos = row_ptr[nrows] - 1;
    
    free(nnz_displs);
}

static double CSR_RowSeg(
	const int seg_len, const int *__restrict col_idx, 
	const double *__restrict val, const double *__restrict x
)
{
	double res = 0.0;
	#pragma omp simd
	for (int index = 0; index < seg_len; index++)
		res += val[index] * x[col_idx[index]];
	return res;
}

static void CSR_IntactRowBlock(
	const int srow, const int erow,
	const int *row_ptr, const int *col_idx, const double *val, 
	const double *__restrict x, double *__restrict y
)
{
	for (int irow = srow; irow < erow; irow++)
    {
        double res = 0.0;
        #pragma omp simd
        for (int index = row_ptr[irow]; index < row_ptr[irow + 1]; index++)
            res += val[index] * x[col_idx[index]];
        y[irow] = res;
    }
}

static void CSRP_SpMV_block(
	const int *row_ptr, const int *col_idx, const double *val, 
    CSRP_blk_info_t blk_info, const int block_id, const double *x, double *y
)
{
	if (blk_info[block_id].first_row == blk_info[block_id].last_row)
	{
		double head_res = CSR_RowSeg(
			blk_info[block_id].nnz_epos - blk_info[block_id].nnz_spos + 1,
			col_idx + blk_info[block_id].nnz_spos,
			val     + blk_info[block_id].nnz_spos,
			x
		);
		
        blk_info[block_id].fr_res = head_res;
        blk_info[block_id].lr_res = 0.0;
	} else {
		// This thread handles segments on more than 1 row
		
		int first_intact_row = blk_info[block_id].first_row;
		int last_intact_row  = blk_info[block_id].last_row;
		
		if (blk_info[block_id].fr_intact == 0) 
		{
			int first_row_epos = row_ptr[blk_info[block_id].first_row + 1];
			blk_info[block_id].fr_res = CSR_RowSeg(
				first_row_epos - blk_info[block_id].nnz_spos,
				col_idx + blk_info[block_id].nnz_spos,
				val     + blk_info[block_id].nnz_spos,
				x
			);
			first_intact_row++;
		}
		
		if (blk_info[block_id].lr_intact == 0) 
		{
			int last_row_spos = row_ptr[blk_info[block_id].last_row];
			blk_info[block_id].lr_res = CSR_RowSeg(
				blk_info[block_id].nnz_epos - last_row_spos + 1,
				col_idx + last_row_spos,
				val     + last_row_spos,
				x
			);
			last_intact_row--;
		}
		
		CSR_IntactRowBlock(
			first_intact_row, last_intact_row + 1,
			row_ptr, col_idx, val, x, y
		);
	}
}

// Perform OpenMP paralleled CSR SpMV using partitioned block information
void CSRP_SpMV(
    const int *row_ptr, const int *col_idx, const double *val, 
    const int nrows, const int nblocks, CSRP_blk_info_t blk_info, 
    const double *x, double *y
)
{
    #pragma omp parallel 
    {
        #pragma omp for 
        for (int i = 0; i < nrows; i++) y[i] = 0.0;
        
        #pragma omp for
        for (int block_id = 0; block_id < nblocks; block_id++)
        {
            CSRP_SpMV_block(
                row_ptr, col_idx, val, 
                blk_info, block_id, x, y
            );
        }
    }
	
	// Accumulate the results for the threads that shared the same row
	for (int block_id = 0; block_id < nblocks; block_id++)
	{
		if (blk_info[block_id].fr_intact == 0)
			y[blk_info[block_id].first_row] += blk_info[block_id].fr_res;
		
		if (blk_info[block_id].lr_intact == 0)
			y[blk_info[block_id].last_row]  += blk_info[block_id].lr_res;
	}
}
