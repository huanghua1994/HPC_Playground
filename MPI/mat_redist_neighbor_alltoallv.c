#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

struct mat_redist_info
{
    MPI_Comm comm;          // MPI communicator
    int      nproc;         // Number of MPI processes in comm
    int      rank;          // Rank of this MPI process in comm
    int      src_srow;      // The starting row of this process's source matrix block
    int      src_scol;      // Number of rows of this process's source matrix block
    int      src_nrow;      // The starting columns of this process's source matrix block
    int      src_ncol;      // Number of columns of this process's source matrix block
    int      req_srow;      // The starting row this process requires
    int      req_scol;      // Number of rows this process requires
    int      req_nrow;      // The starting columns this process requires
    int      req_ncol;      // Number of columns this process requires
    int      n_proc_send;   // Number of processes this process needs to send its original block to
    int      n_proc_recv;   // Number of processes this process needs to receive its required block from
    int      *send_ranks;   // Size n_proc_send, MPI ranks this process need to send a block to 
    int      *send_sizes;   // Size n_proc_send, sizes of blocks this process need to send
    int      *send_displs;  // Size n_proc_send+1, send block displacements in send_buf
    int      *sblk_sizes;   // Size n_proc_send*4, each row describes a send block's srow, scol, nrow, ncol
    int      *recv_ranks;   // Size n_proc_recv, MPI ranks this process need to receive a block from
    int      *recv_sizes;   // Size n_proc_recv, sizes of blocks this process need to receive
    int      *recv_displs;  // Size n_proc_recv+1, receive block displacements in recv_buf
    int      *rblk_sizes;   // Size n_proc_recv*4, each row describes a receive block's srow, scol, nrow, ncol
    double   *send_buf;     // Send buffer
    double   *recv_buf;     // Receive buffer
};
typedef struct mat_redist_info  mat_redist_info_s;
typedef struct mat_redist_info* mat_redist_info_t;

static void calc_seg_intersection(
    int s0, int e0, int s1, int e1, 
    int *is_intersect, int *is, int *ie
)
{
    if (s0 > s1)
    {
        int swap;
        swap = s0; s0 = s1; s1 = swap;
        swap = e0; e0 = e1; e1 = swap;
    }
    if (s1 > e0)
    {
        *is_intersect = 0;
        *is = -1;
        *ie = -1;
        return;
    }
    *is_intersect = 1;
    *is = s1;
    *ie = (e0 < e1) ? e0 : e1;
}

static void calc_rect_intersection(
    int xs0, int xe0, int ys0, int ye0,
    int xs1, int xe1, int ys1, int ye1,
    int *is_intersect, int *ixs, int *ixe, int *iys, int *iye
)
{
    calc_seg_intersection(xs0, xe0, xs1, xe1, is_intersect, ixs, ixe);
    if (*is_intersect == 0) return;
    calc_seg_intersection(ys0, ye0, ys1, ye1, is_intersect, iys, iye);
}

static void copy_dbl_mat_blk(
    double *dst, const int ldd, const double *src, const int lds, 
    const int nrow, const int ncol
)
{
    for (int irow = 0; irow < nrow; irow++)
        memcpy(dst + irow * ldd, src + irow * lds, sizeof(double) * ncol);
} 

// Set up a mat_redist_info_s for redistributing a 2D partitioned matrix
// Note: the source blocks of any two processes should not overlap with each other
// Input parameters:
//   src_s{row, col} : The starting row / column of this process's source matrix block
//   src_n{row, col} : Number of rows / columns of this process's source matrix block
//   req_s{row, col} : The starting row / column this process requires
//   req_n{row, col} : Number of rows / columns this process requires
//   comm            : MPI communicator
// Output parameter:
//   *info_ : Initialized mat_redist_info_t
void mat_redist_info_build(
    const int src_srow, const int src_scol, const int src_nrow, const int src_ncol, 
    const int req_srow, const int req_scol, const int req_nrow, const int req_ncol,
    MPI_Comm comm, mat_redist_info_t *info_
)
{
    mat_redist_info_t info = (mat_redist_info_t) malloc(sizeof(mat_redist_info_s));

    // Set up basic MPI and source block info
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);
    //info->comm  = comm;
    info->rank  = rank;
    info->nproc = nproc;
    
    info->src_srow = src_srow;
    info->src_nrow = src_nrow;
    info->src_scol = src_scol;
    info->src_ncol = src_ncol;
    info->req_srow = req_srow;
    info->req_nrow = req_nrow;
    info->req_scol = req_scol;
    info->req_ncol = req_ncol;

    // Gather all processes' source and required block info
    int src_erow = src_srow + src_nrow - 1;
    int src_ecol = src_scol + src_ncol - 1;
    int req_erow = req_srow + req_nrow - 1;
    int req_ecol = req_scol + req_ncol - 1;
    int my_src_req_info[8] = {
        src_srow, src_scol, src_erow, src_ecol, 
        req_srow, req_scol, req_erow, req_ecol
    };
    int *all_src_req_info = (int*) malloc(sizeof(int) * 8 * nproc);
    MPI_Allgather(my_src_req_info, 8, MPI_INT, all_src_req_info, 8, MPI_INT, comm);

    // Calculate send_info
    int send_cnt = 0, n_proc_send = 0;
    int is_intersect, int_srow, int_erow, int_scol, int_ecol;
    int *send_info0 = (int*) malloc(sizeof(int) * 6 * nproc);
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        int *i_req_info = all_src_req_info + iproc * 8 + 4;
        int i_req_srow = i_req_info[0];
        int i_req_scol = i_req_info[1];
        int i_req_erow = i_req_info[2];
        int i_req_ecol = i_req_info[3];
        calc_rect_intersection(
            src_srow, src_erow, src_scol, src_ecol,
            i_req_srow, i_req_erow, i_req_scol, i_req_ecol,
            &is_intersect, &int_srow, &int_erow, &int_scol, &int_ecol
        );
        if (is_intersect)
        {
            int *send_info0_i = send_info0 + n_proc_send * 6;
            send_info0_i[0] = int_srow;
            send_info0_i[1] = int_scol;
            send_info0_i[2] = int_erow - int_srow + 1;
            send_info0_i[3] = int_ecol - int_scol + 1;
            send_info0_i[4] = iproc;
            send_info0_i[5] = send_cnt;
            n_proc_send++;
            send_cnt += send_info0_i[2] * send_info0_i[3];
        }
    }  // End of iproc loop
    int    *send_ranks  = (int*)    malloc(sizeof(int) * n_proc_send);
    int    *send_sizes  = (int*)    malloc(sizeof(int) * n_proc_send);
    int    *send_displs = (int*)    malloc(sizeof(int) * (n_proc_send + 1));
    int    *sblk_sizes  = (int*)    malloc(sizeof(int) * n_proc_send * 4);
    double *send_buf    = (double*) malloc(sizeof(double) * send_cnt);
    if (send_ranks == NULL || send_sizes == NULL || send_displs == NULL || sblk_sizes == NULL || send_buf == NULL)
    {
        fprintf(stderr, "[ERROR] Failed to allocate send_info (size %d) or send_buf (size %d)\n", 7 * n_proc_send, send_cnt);
        free(info);
        *info_ = NULL;
        return;
    }
    for (int i = 0; i < n_proc_send; i++)
    {
        int *send_info0_i = send_info0 + i * 6;
        int *sblk_size_i  = sblk_sizes + i * 4;
        sblk_size_i[0] = send_info0_i[0];
        sblk_size_i[1] = send_info0_i[1];
        sblk_size_i[2] = send_info0_i[2];
        sblk_size_i[3] = send_info0_i[3];
        send_ranks[i]  = send_info0_i[4];
        send_displs[i] = send_info0_i[5];
        send_sizes[i]  = sblk_size_i[2] * sblk_size_i[3];
    }
    send_displs[n_proc_send] = send_cnt;
    info->n_proc_send = n_proc_send;
    info->send_ranks  = send_ranks;
    info->send_sizes  = send_sizes;
    info->send_displs = send_displs;
    info->sblk_sizes  = sblk_sizes;
    info->send_buf    = send_buf;
    free(send_info0);

    // Calculate recv_info
    int recv_cnt = 0, n_proc_recv = 0;
    int *recv_info0 = (int*) malloc(sizeof(int) * 6 * nproc);
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        int *i_src_info = all_src_req_info + iproc * 8;
        int i_src_srow = i_src_info[0];
        int i_src_scol = i_src_info[1];
        int i_src_erow = i_src_info[2];
        int i_src_ecol = i_src_info[3];
        calc_rect_intersection(
            req_srow, req_erow, req_scol, req_ecol,
            i_src_srow, i_src_erow, i_src_scol, i_src_ecol,
            &is_intersect, &int_srow, &int_erow, &int_scol, &int_ecol
        );
        if (is_intersect)
        {
            int *recv_info0_i = recv_info0 + n_proc_recv * 6;
            recv_info0_i[0] = int_srow;
            recv_info0_i[1] = int_scol;
            recv_info0_i[2] = int_erow - int_srow + 1;
            recv_info0_i[3] = int_ecol - int_scol + 1;
            recv_info0_i[4] = iproc;
            recv_info0_i[5] = recv_cnt;
            n_proc_recv++;
            recv_cnt += recv_info0_i[2] * recv_info0_i[3];
        }
    }  // End of iproc loop
    int    *recv_ranks  = (int*)    malloc(sizeof(int) * n_proc_recv);
    int    *recv_sizes  = (int*)    malloc(sizeof(int) * n_proc_recv);
    int    *recv_displs = (int*)    malloc(sizeof(int) * (n_proc_recv + 1));
    int    *rblk_sizes  = (int*)    malloc(sizeof(int) * n_proc_recv * 4);
    double *recv_buf = (double*) malloc(sizeof(double) * recv_cnt);
    if (recv_ranks == NULL || recv_sizes == NULL || recv_displs == NULL || rblk_sizes == NULL || recv_buf == NULL)
    {
        fprintf(stderr, "[ERROR] Failed to allocate recv_info (size %d) or recv_buf (size %d)\n", 7 * n_proc_recv, recv_cnt);
        free(info);
        *info_ = NULL;
        return;
    }
    for (int i = 0; i < n_proc_recv; i++)
    {
        int *recv_info0_i = recv_info0 + i * 6;
        int *rblk_size_i  = rblk_sizes + i * 4;
        rblk_size_i[0] = recv_info0_i[0];
        rblk_size_i[1] = recv_info0_i[1];
        rblk_size_i[2] = recv_info0_i[2];
        rblk_size_i[3] = recv_info0_i[3];
        recv_ranks[i]  = recv_info0_i[4];
        recv_displs[i] = recv_info0_i[5];
        recv_sizes[i]  = rblk_size_i[2] * rblk_size_i[3];
    }
    recv_displs[n_proc_recv] = recv_cnt;
    info->n_proc_recv = n_proc_recv;
    info->recv_ranks  = recv_ranks;
    info->recv_sizes  = recv_sizes;
    info->recv_displs = recv_displs;
    info->rblk_sizes  = rblk_sizes;
    info->recv_buf    = recv_buf;
    free(recv_info0);

    // Build a new communicator with graph info
    int reorder = 0;
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    MPI_Dist_graph_create_adjacent(comm, n_proc_recv, recv_ranks, MPI_UNWEIGHTED, n_proc_send, send_ranks, MPI_UNWEIGHTED, mpi_info, reorder, &info->comm);
    MPI_Info_free(&mpi_info);

    free(all_src_req_info);
    *info_ = info;
}

// Destroy a mat_redist_info_s
void mat_redist_info_destroy(mat_redist_info_t info)
{
    if (info == NULL) return;
    free(info->send_ranks);
    free(info->send_sizes);
    free(info->send_displs);
    free(info->sblk_sizes);
    free(info->recv_ranks);
    free(info->recv_sizes);
    free(info->recv_displs);
    free(info->rblk_sizes);
    free(info->send_buf);
    free(info->recv_buf);
    free(info);
}

// Perform matrix data redistribution
// Input parameters:
//   info : Initialized mat_redist_info_t
//   src_blk : Source matrix block of this process
//   src_ld  : Leading dimension of src_blk
//   dst_ld  : Leading dimension of dst_blk
// Output parameter:
//   dst_blk : Destination (required) matrix block of this process
void mat_redist_exec(
    mat_redist_info_t info, const double *src_blk, const int src_ld, 
    double *dst_blk, const int dst_ld
)
{
    if (info == NULL)
    {
        fprintf(stderr, "[ERROR] mat_redist_info_t == NULL\n");
        return;
    }

    // Pack the send_buf
    int    src_srow     = info->src_srow;
    int    src_scol     = info->src_scol;
    int    n_proc_send  = info->n_proc_send;
    int    *send_sizes  = info->send_sizes;
    int    *send_displs = info->send_displs;
    int    *sblk_sizes  = info->sblk_sizes;
    double *send_buf    = info->send_buf;
    for (int isend = 0; isend < n_proc_send; isend++)
    {
        int *i_sblk_size = sblk_sizes + isend * 4;
        int i_send_srow = i_sblk_size[0];
        int i_send_scol = i_sblk_size[1];
        int i_send_nrow = i_sblk_size[2];
        int i_send_ncol = i_sblk_size[3];
        int local_srow  = i_send_srow - src_srow;
        int local_scol  = i_send_scol - src_scol;
        double *i_send_buf = send_buf + send_displs[isend];
        const double *i_send_src = src_blk + local_srow * src_ld + local_scol;
        copy_dbl_mat_blk(i_send_buf, i_send_ncol, i_send_src, src_ld, i_send_nrow, i_send_ncol);
    }  // End of isend loop

    // Redistribute data using MPI_Neighbor_alltoallv
    int    *recv_sizes  = info->recv_sizes;
    int    *recv_displs = info->recv_displs;
    double *recv_buf    = info->recv_buf;
    MPI_Neighbor_alltoallv(
        send_buf, send_sizes, send_displs, MPI_DOUBLE, 
        recv_buf, recv_sizes, recv_displs, MPI_DOUBLE, info->comm
    );

    // Repack received blocks
    int req_srow    = info->req_srow;
    int req_scol    = info->req_scol;
    int n_proc_recv = info->n_proc_recv;
    int *rblk_sizes = info->rblk_sizes;
    for (int irecv = 0; irecv < n_proc_recv; irecv++)
    {
        int *i_rblk_size = rblk_sizes + irecv * 4;
        int i_recv_srow = i_rblk_size[0];
        int i_recv_scol = i_rblk_size[1];
        int i_recv_nrow = i_rblk_size[2];
        int i_recv_ncol = i_rblk_size[3];
        int local_srow  = i_recv_srow - req_srow;
        int local_scol  = i_recv_scol - req_scol;
        double *i_recv_buf = recv_buf + recv_displs[irecv];
        double *i_recv_dst = dst_blk + local_srow * dst_ld + local_scol;
        copy_dbl_mat_blk(i_recv_dst, dst_ld, i_recv_buf, i_recv_ncol, i_recv_nrow, i_recv_ncol);
    }  // End of recv_cnt loop
}

// ====================== Below are test driver ======================

#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

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
    calc_block_spos_len(glb_nrow, nproc_row, rank_row, &src_srow, &src_nrow);
    calc_block_spos_len(glb_ncol, nproc_col, rank_col, &src_scol, &src_ncol);
    double *src_blk = (double*) malloc(sizeof(double) * src_nrow * src_ncol);
    for (int irow = 0; irow < src_nrow; irow++)
    {
        for (int icol = 0; icol < src_ncol; icol++)
            src_blk[irow * src_ncol + icol] = (double) ((src_srow + irow) * glb_ncol + (src_scol + icol));
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
    mat_redist_info_t redist_info;
    mat_redist_info_build(
        src_srow, src_scol, src_nrow, src_ncol,
        req_srow, req_scol, req_nrow, req_ncol,
        MPI_COMM_WORLD, &redist_info
    );
    MPI_Barrier(MPI_COMM_WORLD);

    // Get required block and compare the results
    double *dst_blk = (double*) malloc(sizeof(double) * req_nrow * req_ncol);
    double st, et, ut = 0.0;
    // Warm up
    mat_redist_exec(redist_info, src_blk, src_ncol, dst_blk, req_ncol);
    // Time it
    int ntest = 10;
    for (int i = 0; i < ntest; i++)
    {
        st = MPI_Wtime();
        mat_redist_exec(redist_info, src_blk, src_ncol, dst_blk, req_ncol);
        et = MPI_Wtime();
        ut += et - st;
    }
    if (rank == 0) printf("%d tests mat_redist_exec used %.3lf sec\n", ntest, ut);
    int err_cnt = 0;
    for (int irow = 0; irow < req_nrow; irow++)
    {
        for (int icol = 0; icol < req_ncol; icol++)
        {
            double expected_val = (double) ((req_srow + irow) * glb_ncol + (req_scol + icol));
            double received_val = dst_blk[irow * req_ncol + icol];
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
    free(src_blk);
    mat_redist_info_destroy(redist_info);
    MPI_Finalize();
    return 0;
}
