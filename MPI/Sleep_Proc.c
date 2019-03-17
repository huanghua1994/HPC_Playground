#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>

#define EXEC_RANK 4

int main(int argc, char **argv)
{
    int nprocs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    double st, et0, et1;
    MPI_Request req;
    MPI_Status status;
    int flag, res = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    st = MPI_Wtime();
    
    if (my_rank < EXEC_RANK) 
    {
        // Working process
        // Let's waste some time
        int *a = (int *) malloc(sizeof(int) * 1919810);
        int *b = (int *) malloc(sizeof(int) * 1919810);
        assert(a != NULL && b != NULL);
        memset(a, 0, sizeof(int) * 1919810);
        memset(b, 0, sizeof(int) * 1919810);
        for (int i = 1; i < 1919810; i++)
        {
            a[i] += a[i - 1] * b[i];
            b[i] += a[i] * b[i - 1];
        }
        res = 1;
        for (int i = 0; i < 1919810; i++) res += b[i];
        free(a);
        free(b);
        et0 = MPI_Wtime();
        
        MPI_Ibarrier(MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &status);
        et1 = MPI_Wtime();
    } else {  
        // Sleeping process
        et0 = MPI_Wtime();
        MPI_Ibarrier(MPI_COMM_WORLD, &req);
        flag = 0;
        while (flag == 0)
        {
            usleep(1000);  // Sleep 10 milliseconds 
            MPI_Test(&req, &flag, &status);
        }
        et1 = MPI_Wtime();
    }
    
    printf("MPI Proc %2d: result = %d, work/total time = %lf, %lf\n", my_rank, res, et0 - st, et1 - st);
    
    MPI_Finalize();
    return 0;
}