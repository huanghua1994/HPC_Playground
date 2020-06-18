#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#define EXEC_RANK 8

int main(int argc, char **argv)
{
    int nprocs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    MPI_Win  mpi_win;
    MPI_Info mpi_info;
    int *buffer = (int*) malloc(sizeof(int) * nprocs);
    memset(buffer, 0, sizeof(int) * nprocs);
    buffer[my_rank] = my_rank;
    
    MPI_Info_create(&mpi_info);
    MPI_Win_create(buffer, nprocs * 4, 4, mpi_info, MPI_COMM_WORLD, &mpi_win);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // MPI RMA part
    MPI_Win_lock_all(0, mpi_win);
    if (my_rank < EXEC_RANK)
    {
        for (int target_rank = 0; target_rank < nprocs; target_rank++)
        {
            if (target_rank == my_rank) continue;
            MPI_Get(buffer + target_rank, 1, MPI_INT, target_rank, target_rank, 1, MPI_INT, mpi_win);
        }
        MPI_Win_flush_all(mpi_win);
    }
    MPI_Win_unlock_all(mpi_win);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (my_rank < EXEC_RANK)
    {
        int correct = 1;
        for (int i = 0; i < nprocs; i++)
            if (buffer[i] != i) correct = 0;
        printf("MPI %d, result correct = %d\n", my_rank, correct);
    }
    
    MPI_Win_free(&mpi_win);
    free(buffer);
    
    MPI_Finalize();
}
