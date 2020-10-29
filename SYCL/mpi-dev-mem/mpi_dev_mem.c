#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <mpi.h>

void MPI_Init_wrapper(int *argc, char ***argv)
{
    MPI_Init(argc, argv);
}

void MPI_Comm_world_size_rank(int *n_proc, int *my_rank)
{
    MPI_Comm_size(MPI_COMM_WORLD, n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, my_rank);
}

void MPI_Finalize_wrapper()
{
    MPI_Finalize();
}

void MPI_test_dev_mem_recv(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec1)
{
    int next_rank = (my_rank + 1) % n_proc;
    int prev_rank = (my_rank == 0) ? (n_proc - 1) : my_rank - 1;
    MPI_Request req;
    MPI_Isend(d_vec0, vec_len, MPI_INT, next_rank, 42, MPI_COMM_WORLD, &req);
    MPI_Recv(d_vec1, vec_len, MPI_INT, prev_rank, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void MPI_test_dev_mem_put(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec2)
{
    int next_rank = (my_rank + 1) % n_proc;
    int prev_rank = (my_rank == 0) ? (n_proc - 1) : my_rank - 1;
    MPI_Win mpi_win;
    MPI_Win_create(d_vec2, sizeof(int) * vec_len, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &mpi_win);
    MPI_Win_lock(MPI_LOCK_SHARED, prev_rank, 0, mpi_win);
    MPI_Put(
        d_vec0, vec_len, MPI_INT, prev_rank, 
        0,      vec_len, MPI_INT, mpi_win
    );
    MPI_Win_unlock(prev_rank, mpi_win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&mpi_win);
}
