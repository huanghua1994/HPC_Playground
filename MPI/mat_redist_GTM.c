#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#include "GTMatrix.h"
#include "utils.h"

void calc_block_spos_len(
    const int len, const int nblk, const int iblk,
    int *blk_spos, int *blk_len
)
{
    if (iblk < 0 || iblk > nblk)
    {
        *blk_spos = -1;
        *blk_len  = 0;
        return;
    }
    int rem = len % nblk;
    int bs0 = len / nblk;
    int bs1 = bs0 + 1;
    if (iblk < rem) 
    {
        *blk_spos = bs1 * iblk;
        *blk_len  = bs1;
    } else {
        *blk_spos = bs0 * iblk + rem;
        *blk_len  = bs0;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // Set up a Cartesian process grid (not really need the communicator)
    int nproc, rank, dims[2] = {0, 0};
    int nproc_row, nproc_col, rank_row, rank_col;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Dims_create(nproc, 2, dims);
    nproc_row = dims[0];
    nproc_col = dims[1];
    rank_row  = rank / nproc_col;
    rank_col  = rank % nproc_col;

    // Initialize local source block
    int glb_nrow, glb_ncol;
    if (argc >= 2) glb_nrow = atoi(argv[1]);
    if (argc >= 3) glb_ncol = atoi(argv[2]);
    if (glb_nrow > 10000 || glb_nrow < 1) glb_nrow = 100;
    if (glb_ncol > 10000 || glb_ncol < 1) glb_ncol = 100;
    if (rank == 0) printf("Source matrix global size = %d * %d\n", glb_nrow, glb_ncol);
    int src_srow, src_scol, src_nrow, src_ncol;
    int *row_displs = (int*) malloc(sizeof(int) * (nproc_row + 1));
    int *col_displs = (int*) malloc(sizeof(int) * (nproc_col + 1));
    for (int i = 0; i < nproc_row; i++)
    {
        calc_block_spos_len(glb_nrow, nproc_row, i, &src_srow, &src_nrow);
        row_displs[i] = src_srow;
    }
    row_displs[nproc_row] = glb_nrow;
    for (int i = 0; i < nproc_col; i++)
    {
        calc_block_spos_len(glb_ncol, nproc_col, i, &src_scol, &src_ncol);
        col_displs[i] = src_scol;
    }
    col_displs[nproc_col] = glb_ncol;
    calc_block_spos_len(glb_nrow, nproc_row, rank_row, &src_srow, &src_nrow);
    calc_block_spos_len(glb_ncol, nproc_col, rank_col, &src_scol, &src_ncol);
    GTMatrix_t gtm;
    GTM_create(
        &gtm, MPI_COMM_WORLD, MPI_DOUBLE, sizeof(double), 
        rank, glb_nrow, glb_ncol, nproc_row, nproc_col, row_displs, col_displs
    );

    double *src_blk = (double*) gtm->mat_block;
    int src_ld = gtm->ld_local;
    for (int irow = 0; irow < src_nrow; irow++)
    {
        for (int icol = 0; icol < src_ncol; icol++)
            src_blk[irow * src_ld + icol] = (double) ((src_srow + irow) * glb_ncol + (src_scol + icol));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Generate a random range for local required block and set up a mat_redist_info_t
    int req_srow, req_scol, req_erow, req_ecol, req_nrow, req_ncol;
    srand(time(NULL) + rank);
    req_srow = rand() % glb_nrow;
    req_scol = rand() % glb_ncol;
    req_erow = rand() % glb_nrow;
    req_ecol = rand() % glb_ncol;
    if (req_srow > req_erow)
    {
        int tmp  = req_erow;
        req_erow = req_srow;
        req_srow = tmp;
    }
    if (req_scol > req_ecol)
    {
        int tmp  = req_ecol;
        req_ecol = req_scol;
        req_scol = tmp;
    }
    req_nrow = req_erow - req_srow + 1;
    req_ncol = req_ecol - req_scol + 1;
    MPI_Barrier(MPI_COMM_WORLD);

    // Get required block and compare the results
    int dst_ld = req_ncol;
    double *dst_blk = (double*) malloc(sizeof(double) * req_nrow * req_ncol);
    double st, et, ut = 0.0;
    // Warm up
    GTM_startBatchGet(gtm);
    GTM_addGetBlockRequest(gtm, req_srow, req_nrow, req_scol, req_ncol, dst_blk, dst_ld);
    GTM_execBatchGet(gtm);
    GTM_stopBatchGet(gtm);
    MPI_Barrier(MPI_COMM_WORLD);
    // Time it
    int ntest = 10;
    for (int i = 0; i < ntest; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = get_wtime_sec();
        GTM_startBatchGet(gtm);
        GTM_addGetBlockRequest(gtm, req_srow, req_nrow, req_scol, req_ncol, dst_blk, dst_ld);
        GTM_execBatchGet(gtm);
        GTM_stopBatchGet(gtm);
        MPI_Barrier(MPI_COMM_WORLD);
        et = get_wtime_sec();
        ut += et - st;
    }
    if (rank == 0) printf("%d tests GTM_execBatchGet used %.3lf sec\n", ntest, ut);
    int err_cnt = 0;
    for (int irow = 0; irow < req_nrow; irow++)
    {
        for (int icol = 0; icol < req_ncol; icol++)
        {
            double expected_val = (double) ((req_srow + irow) * glb_ncol + (req_scol + icol));
            double received_val = dst_blk[irow * dst_ld + icol];
            if (fabs(received_val - expected_val) > 1e-10)
            {
                err_cnt++;
                if (err_cnt <= 3)
                fprintf(
                    stderr, "[ERROR] Rank %d dst_blk(%d, %d) expected %.1lf, received %.1lf\n",
                    rank, irow, icol, expected_val, received_val
                );
            }
        }
    }
    printf(
        "Rank %3d req_block: (%4d : %4d, %4d : %4d), has %d error(s)\n",
        rank, req_srow, req_erow, req_scol, req_ecol, err_cnt
    );

    free(dst_blk);
    GTM_destroy(gtm);
    MPI_Finalize();
    return 0;
}
