#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

struct mat_redist_info
{
    int          unit_size;         // Size (bytes) of matrix element data type
    int          n_proc_get;        // Number of processes this process needs to get its required block from
    int          *get_ranks;        // Size n_proc_get, MPI ranks this process need to get a block from
    size_t       *dst_blk_displs;   // Size n_proc_get, displacements of src_blk_types[i] on its src_blk
    MPI_Aint     *src_blk_displs;   // Size n_proc_get, displacements of dst_blk_types[i] on its dst_blk
    MPI_Win      win;               // MPI window
    MPI_Datatype *src_blk_types;    // Size n_proc_get, MPI DDT for source blocks on remote processes
    MPI_Datatype *dst_blk_types;    // Size n_proc_get, MPI DDT for destination blocks on this process
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


// Set up a mat_redist_info_s for redistributing a 2D partitioned matrix
// Note: the source blocks of any two processes should not overlap with each other
// Input parameters:
//   src_s{row, col} : The starting row / column of this process's source matrix block
//   src_n{row, col} : Number of rows / columns of this process's source matrix block
//   req_s{row, col} : The starting row / column this process requires
//   req_n{row, col} : Number of rows / columns this process requires
//   unit_size       : Size (bytes) of matrix element data type
//   src_{blk, ld}   : Source matrix block of this process and its leading dimension
//   dst_ld          : Leading dimension of the buffer for storing the required matrix block
//   comm            : MPI communicator
// Output parameter:
//   *info_ : Initialized mat_redist_info_t
void mat_redist_info_build(
    const int src_srow, const int src_scol, const int src_nrow, const int src_ncol, 
    const int req_srow, const int req_scol, const int req_nrow, const int req_ncol,
    const int unit_size, void *src_blk, const int src_ld, const int dst_ld, 
    MPI_Comm comm, mat_redist_info_t *info_
)
{
    mat_redist_info_t info = (mat_redist_info_t) malloc(sizeof(mat_redist_info_s));

    // Set up basic MPI and copy source / required block info
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    // Gather all processes' source block info
    int src_erow = src_srow + src_nrow - 1;
    int src_ecol = src_scol + src_ncol - 1;
    int my_src_info[5] = {src_srow, src_scol, src_erow, src_ecol, src_ld};
    int *all_src_info = (int*) malloc(sizeof(int) * 5 * nproc);
    MPI_Allgather(my_src_info, 5, MPI_INT, all_src_info, 5, MPI_INT, comm);

    // Find the processes that this process need to get a block from
    int n_proc_get = 0;
    int *recv_info = (int*) malloc(sizeof(int) * 8 * nproc);
    int req_erow = req_srow + req_nrow - 1;
    int req_ecol = req_scol + req_ncol - 1;
    int is_intersect;
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        int *i_src_info = all_src_info + iproc * 5;
        int i_src_srow = i_src_info[0];
        int i_src_scol = i_src_info[1];
        int i_src_erow = i_src_info[2];
        int i_src_ecol = i_src_info[3];
        int i_src_ld   = i_src_info[4];
        int int_srow, int_scol, int_erow, int_ecol;
        calc_rect_intersection(
            req_srow, req_erow, req_scol, req_ecol,
            i_src_srow, i_src_erow, i_src_scol, i_src_ecol,
            &is_intersect, &int_srow, &int_erow, &int_scol, &int_ecol
        );
        if (is_intersect)
        {
            int *recv_info_i = recv_info + n_proc_get * 8;
            recv_info_i[0] = int_srow;
            recv_info_i[1] = int_scol;
            recv_info_i[2] = int_erow - int_srow + 1;
            recv_info_i[3] = int_ecol - int_scol + 1;
            recv_info_i[4] = i_src_srow;
            recv_info_i[5] = i_src_scol;
            recv_info_i[6] = i_src_ld;
            recv_info_i[7] = iproc;
            n_proc_get++;
        }
    }  // End of iproc loop

    // Create MPI window and MPI DDT
    int          *get_ranks      = (int*)          malloc(sizeof(int)          * n_proc_get);
    size_t       *dst_blk_displs = (size_t*)       malloc(sizeof(size_t)       * n_proc_get);
    MPI_Aint     *src_blk_displs = (MPI_Aint*)     malloc(sizeof(MPI_Aint)     * n_proc_get);
    MPI_Datatype *src_blk_types  = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * n_proc_get);
    MPI_Datatype *dst_blk_types  = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * n_proc_get);
    for (int iproc = 0; iproc < n_proc_get; iproc++)
    {
        int *recv_info_i = recv_info + iproc * 8;
        int int_srow   = recv_info_i[0];
        int int_scol   = recv_info_i[1];
        int int_nrow   = recv_info_i[2];
        int int_ncol   = recv_info_i[3];
        int i_src_srow = recv_info_i[4];
        int i_src_scol = recv_info_i[5];
        int i_src_ld   = recv_info_i[6];
        get_ranks[iproc]   = recv_info_i[7];
        int i_src_rel_srow = int_srow - i_src_srow;
        int i_src_rel_scol = int_scol - i_src_scol;
        int i_dst_rel_srow = int_srow - req_srow;
        int i_dst_rel_scol = int_scol - req_scol;
        src_blk_displs[iproc] = (MPI_Aint) i_src_rel_srow * (MPI_Aint) i_src_ld + (MPI_Aint) i_src_rel_scol;
        dst_blk_displs[iproc] = (size_t)   i_dst_rel_srow * (size_t)     dst_ld + (size_t)   i_dst_rel_scol;
        MPI_Type_vector(int_nrow, int_ncol, i_src_ld, MPI_DOUBLE, src_blk_types + iproc);
        MPI_Type_vector(int_nrow, int_ncol,   dst_ld, MPI_DOUBLE, dst_blk_types + iproc);
        MPI_Type_commit(src_blk_types + iproc);
        MPI_Type_commit(dst_blk_types + iproc);
    }  // End of iproc loop
    info->unit_size      = unit_size;
    info->n_proc_get     = n_proc_get;
    info->get_ranks      = get_ranks;
    info->src_blk_displs = src_blk_displs;
    info->dst_blk_displs = dst_blk_displs;
    info->src_blk_types  = src_blk_types;
    info->dst_blk_types  = dst_blk_types;
    MPI_Info mpi_info;
    MPI_Aint my_blk_size = (MPI_Aint) src_nrow * (MPI_Aint) src_ld;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(src_blk, my_blk_size, unit_size, mpi_info, comm, &info->win);
    MPI_Info_free(&mpi_info);

    free(recv_info);
    free(all_src_info);
    *info_ = info;
}

// Destroy a mat_redist_info_s
void mat_redist_info_destroy(mat_redist_info_t info)
{
    if (info == NULL) return;
    free(info->get_ranks);
    free(info->src_blk_displs);
    free(info->dst_blk_displs);
    for (int i = 0; i < info->n_proc_get; i++)
    {
        MPI_Type_free(info->src_blk_types + i);
        MPI_Type_free(info->dst_blk_types + i);
    }
    MPI_Win_free(&info->win);
    free(info);
}

// Perform matrix data redistribution
// Input parameters:
//   info : Initialized mat_redist_info_t
// Output parameter:
//   dst_blk : Destination (required) matrix block of this process
void mat_redist_exec(mat_redist_info_t info, void *dst_blk)
{
    if (info == NULL)
    {
        fprintf(stderr, "[ERROR] mat_redist_info_t == NULL\n");
        return;
    }

    int unit_size  = info->unit_size;
    int n_proc_get = info->n_proc_get;
    int *get_ranks = info->get_ranks;
    size_t   *dst_blk_displs = info->dst_blk_displs;
    MPI_Aint *src_blk_displs = info->src_blk_displs;
    MPI_Datatype *src_blk_types = info->src_blk_types;
    MPI_Datatype *dst_blk_types = info->dst_blk_types;
    MPI_Win_fence(0, info->win);
    for (int iget = 0; iget < n_proc_get; iget++)
    {
        char *dst_ptr = (char*) dst_blk + unit_size * dst_blk_displs[iget];
        MPI_Get(
            (void*) dst_ptr,      1, dst_blk_types[iget], get_ranks[iget], 
            src_blk_displs[iget], 1, src_blk_types[iget], info->win
        );
    }
    MPI_Win_fence(0, info->win);
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
    int glb_nrow = 100, glb_ncol = 100;
    if (argc >= 2) glb_nrow = atoi(argv[1]);
    if (argc >= 3) glb_ncol = atoi(argv[2]);
    if (glb_nrow > 10000 || glb_nrow < 1) glb_nrow = 100;
    if (glb_ncol > 10000 || glb_ncol < 1) glb_ncol = 100;
    if (rank == 0) printf("Source matrix global size = %d * %d\n", glb_nrow, glb_ncol);
    int src_srow, src_scol, src_nrow, src_ncol;
    calc_block_spos_len(glb_nrow, nproc_row, rank_row, &src_srow, &src_nrow);
    calc_block_spos_len(glb_ncol, nproc_col, rank_col, &src_scol, &src_ncol);
    double *src_blk = (double*) malloc(sizeof(double) * src_nrow * src_ncol);
    int src_ld = src_ncol;
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
    int dst_ld = req_ncol;
    MPI_Barrier(MPI_COMM_WORLD);

    mat_redist_info_t redist_info;
    mat_redist_info_build(
        src_srow, src_scol, src_nrow, src_ncol,
        req_srow, req_scol, req_nrow, req_ncol,
        sizeof(double), src_blk, src_ld, dst_ld, 
        MPI_COMM_WORLD, &redist_info
    );
    MPI_Barrier(MPI_COMM_WORLD);

    // Get required block and compare the results
    double *dst_blk = (double*) malloc(sizeof(double) * req_nrow * req_ncol);
    double st, et, ut = 0.0;
    // Warm up
    mat_redist_exec(redist_info, dst_blk);
    MPI_Barrier(MPI_COMM_WORLD);
    // Time it
    int ntest = 10;
    for (int i = 0; i < ntest; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        mat_redist_exec(redist_info, dst_blk);
        MPI_Barrier(MPI_COMM_WORLD);
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
    free(src_blk);
    mat_redist_info_destroy(redist_info);
    MPI_Finalize();
    return 0;
}
