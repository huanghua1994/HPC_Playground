#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int nprocs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    MPI_Win  mpi_win;
    MPI_Info mpi_info;
    int *buffer = (int*) malloc(sizeof(int) * 2);
    int *my_counter = buffer;
    
    // Each process create one 4-byte integer to work as a counter
    MPI_Info_create(&mpi_info);
    MPI_Win_create(my_counter, 4, 4, mpi_info, MPI_COMM_WORLD, &mpi_win);
    
    my_counter[0] = my_rank * 100;
    printf("MPI rank %d initial counter = %d\n", my_rank, my_counter[0]);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    int *target_counter = buffer + 1;
    int addend = 1;
    int target_rank = 0; 
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, mpi_win);
    if (my_rank == target_rank)
    {
        target_counter[0] = __sync_fetch_and_add(my_counter, 1);
    } else {
        MPI_Fetch_and_op(&addend, target_counter, MPI_INT, target_rank, 0, MPI_SUM, mpi_win);
    }
    MPI_Win_unlock(target_rank, mpi_win);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("MPI rank %d fetched counter = %d, self counter = %d\n", my_rank, target_counter[0], my_counter[0]);
    
    MPI_Win_free(&mpi_win);
    free(buffer);
    
    MPI_Finalize();
}