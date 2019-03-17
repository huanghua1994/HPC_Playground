#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

// mpiicc acc.c -std=gnu99 -o acc.x

// When using Intel MPI 2017 and more than 16 MPI processes per node,
// blocking_barrier == 1 is likely to trigger a deadlock. It seems 
// that IMPI 2018 and later version fixed this problem.

#define BUFLEN   32
#define ACT_RANK 8

int main(int argc, char **argv)
{
    int nprocs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    MPI_Win  mpi_win;
    MPI_Info mpi_info;
    int *buffer = (int*) malloc(sizeof(int) * BUFLEN);
    int *addend = (int*) malloc(sizeof(int) * BUFLEN);
    memset(buffer, 0, sizeof(int) * BUFLEN);
    for (int i = 0; i < BUFLEN; i++) addend[i] = my_rank;
    
    MPI_Info_create(&mpi_info);
    MPI_Win_create(buffer, nprocs * BUFLEN * 4, 4, mpi_info, MPI_COMM_WORLD, &mpi_win);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (my_rank < ACT_RANK)
    {
        for (int i = 0; i < nprocs; i++)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, i, 0, mpi_win);
            MPI_Accumulate(addend, BUFLEN, MPI_INT, i, 0, BUFLEN, MPI_INT, MPI_SUM, mpi_win);
            MPI_Win_unlock(i, mpi_win);
        }
    }
    
    int blocking_barrier = 0;
    if (blocking_barrier == 1)
    {
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        MPI_Request req;
        MPI_Status  status;
        MPI_Ibarrier(MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &status);
    }
    
    int stdres = (ACT_RANK * (ACT_RANK - 1)) / 2;
    int correct = 1, allcorrect;
    for (int i = 0; i < BUFLEN; i++)
    {
        if (buffer[i] != stdres) 
        {
            printf("Rank %d: error at %d\n", my_rank, i);
            correct = 0;
        }
    }
    MPI_Reduce(&correct, &allcorrect, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    if (my_rank == 0) printf("\nResults correct = %d\n", allcorrect);
    
    MPI_Win_free(&mpi_win);
    free(buffer);
    free(addend);
    
    MPI_Finalize();
}
