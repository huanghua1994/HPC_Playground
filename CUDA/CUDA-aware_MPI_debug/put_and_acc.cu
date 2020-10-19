#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

int main(int argc, char **argv)
{
    int local_rank = atoi(getenv("MPI_LOCALRANKID"));
    CUDA_RT_CALL( cudaSetDevice(local_rank) );

    MPI_Init(&argc, &argv);

    int my_rank, n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int arr_size = 1024;
    if (argc >= 2) arr_size = atoi(argv[1]);
    if (arr_size < 0) arr_size = 1024;
    if (my_rank == 0) printf("arr_size = %d\n", arr_size);

    int send_rank = n_proc - 1;
    int recv_rank = 0;
    if (argc >= 3) send_rank = atoi(argv[2]);
    if (argc >= 4) send_rank = atoi(argv[3]);
    if (my_rank == 0) printf("send_rank, recv_rank = %d, %d\n", send_rank, recv_rank);

    int *dev_arr;
    size_t arr_bytes = sizeof(int) * arr_size;
    CUDA_RT_CALL( cudaMalloc(&dev_arr, arr_bytes) );
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win mpi_win;
    MPI_Win_create(dev_arr, arr_bytes, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &mpi_win);
    MPI_Win_lock_all(0, mpi_win);
    
    if (my_rank == send_rank)
    {
        MPI_Put(dev_arr, arr_size, MPI_INT, recv_rank, 0, arr_size, MPI_INT, mpi_win);
        MPI_Win_flush(recv_rank, mpi_win);
        printf("Rank %d calls MPI_Put done\n", my_rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == send_rank)
    {
        MPI_Accumulate(dev_arr, arr_size, MPI_INT, recv_rank, 0, arr_size, MPI_INT, MPI_SUM, mpi_win);
        MPI_Win_flush(recv_rank, mpi_win);
        printf("Rank %d calls MPI_Accumulate done\n", my_rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_unlock_all(mpi_win);

    CUDA_RT_CALL( cudaFree(dev_arr) );

    MPI_Win_free(&mpi_win);
    MPI_Finalize();
    return 0;
}
