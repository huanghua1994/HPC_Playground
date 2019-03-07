#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>

#define BLK_LEN   (1024 * 1024 * 8)  // 8M elements, 64 MB, takes > 0.005333 seconds to be sent @ 12000 MB/s

// Compile: mpiicc -O3 -xHost Ireduce_stime.c -o Ireduce_stime.x
/* 
Example output on 4 Stampede2 SKX nodes:
MPI Proc  0: start Ireduce at, finish Ireduce at, finish wait at, Ireduce duration = 0.000001, 0.004878, 0.071947, 0.004877
MPI Proc  1: start Ireduce at, finish Ireduce at, finish wait at, Ireduce duration = 0.050132, 0.061673, 0.072652, 0.011541
MPI Proc  2: start Ireduce at, finish Ireduce at, finish wait at, Ireduce duration = 0.050078, 0.060711, 0.071464, 0.010633
MPI Proc  3: start Ireduce at, finish Ireduce at, finish wait at, Ireduce duration = 0.050132, 0.060493, 0.070858, 0.010361
0.004877 < 0.005333, so the communicated data is not sent in Ireduce on rank 0
*/

int main(int argc, char **argv)
{
    int nprocs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    double st0, st1, et0, et1;
    MPI_Request req;
    MPI_Status status;
    int flag, res = 0;
    
    int *a = (int *) malloc(sizeof(int) * BLK_LEN);
    int *b = (int *) malloc(sizeof(int) * BLK_LEN);
    assert(a != NULL && b != NULL);
    memset(a, 0, sizeof(int) * BLK_LEN);
    memset(b, 0, sizeof(int) * BLK_LEN);
    
    MPI_Barrier(MPI_COMM_WORLD);
    st0 = MPI_Wtime();
    if (my_rank > 0) usleep(50000); // If rank > 0, sleep 0.05 seconds
    st1 = MPI_Wtime();
    MPI_Ireduce(a, b, BLK_LEN, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &req);
    et0 = MPI_Wtime();
    MPI_Wait(&req, &status);
    et1 = MPI_Wtime();
    
    printf(
        "MPI Proc %2d: start Ireduce at, finish Ireduce at, finish wait at, Ireduce duration = %lf, %lf, %lf, %lf\n", 
        my_rank, st1 - st0, et0 - st0, et1 - st0, et0 - st1
    );
    
    free(a);
    free(b);
    
    MPI_Finalize();
    return 0;
}