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
    MPI_Init(&argc, &argv);

    int my_rank, n_proc, prev_rank, next_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    prev_rank = (my_rank > 0) ? (my_rank - 1) : (n_proc - 1);
    next_rank = (my_rank + 1) % n_proc;

    int arr_size = 1024;
    if (argc >= 2) arr_size = atoi(argv[1]);
    if (arr_size < 0) arr_size = 1024;
    if (my_rank == 0) printf("arr_size = %d\n", arr_size);

    int shm_my_rank, shm_n_proc;
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_size(shm_comm, &shm_n_proc);
    MPI_Comm_rank(shm_comm, &shm_my_rank);
    MPI_Comm_free(&shm_comm);

    int *dev_arr;
    size_t arr_bytes = sizeof(int) * arr_size;
    CUDA_RT_CALL( cudaSetDevice(shm_my_rank) );
    CUDA_RT_CALL( cudaMalloc(&dev_arr, 4 * arr_bytes) );
    MPI_Barrier(MPI_COMM_WORLD);

    int *prev_recv_ptr = dev_arr + arr_size * 0;
    int *prev_send_ptr = dev_arr + arr_size * 1;
    int *next_send_ptr = dev_arr + arr_size * 2;
    int *next_recv_ptr = dev_arr + arr_size * 3;
    MPI_Sendrecv(
        prev_send_ptr, arr_size, MPI_INT, prev_rank, 0,
        next_recv_ptr, arr_size, MPI_INT, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        next_send_ptr, arr_size, MPI_INT, next_rank, 0,
        prev_recv_ptr, arr_size, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    printf("Rank %d finished 1D halo exchange\n", my_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_RT_CALL( cudaFree(dev_arr) );

    MPI_Finalize();
    return 0;
}