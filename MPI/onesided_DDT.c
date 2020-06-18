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
    int *buffer = (int*) malloc(sizeof(int) * 100);
    memset(buffer, 0, sizeof(int) * 100);
    for (int i = 0; i < 100; i++)
        buffer[i] = my_rank * 100 + i;
    
    MPI_Info_create(&mpi_info);
    MPI_Win_create(buffer, 100 * 4, 4, mpi_info, MPI_COMM_WORLD, &mpi_win);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI RMA part
    MPI_Win_lock_all(0, mpi_win);
    if (my_rank == 0)
    {
        MPI_Datatype src_block_type, dst_block_type;
        MPI_Type_vector(3, 5, 10, MPI_INT, &src_block_type);
        MPI_Type_vector(3, 5,  9, MPI_INT, &dst_block_type);
        MPI_Type_commit(&src_block_type);
        MPI_Type_commit(&dst_block_type);
        // Get rank 1's [3:5, 2:6] (rank 1 is 9 * 9 matrix)
        // and copy to rank 0's [2:4, 2:6] (rank 0 is 10 * 10 matrix)
        MPI_Get(buffer + 11, 1, src_block_type, 1, 
                         21, 1, dst_block_type, mpi_win);
        MPI_Type_free(&dst_block_type);
        MPI_Type_free(&src_block_type);
        MPI_Win_flush_all(mpi_win);
    }
    MPI_Win_unlock_all(mpi_win);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (my_rank == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++) printf("%4d ", buffer[i * 10 + j]);
            printf("\n");
        }
    }
    
    MPI_Win_free(&mpi_win);
    free(buffer);
    
    MPI_Finalize();
}