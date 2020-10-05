#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <mpi.h>
#include <unistd.h>

#include "cuda_proxy.h"
#include "test_kernel.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int send_rank = 1, recv_rank = 0;

    if (argc >= 2) send_rank = atoi(argv[1]);
    if (argc >= 3) recv_rank = atoi(argv[2]);

    // Set up MPI shared memory communicator to get shared memory rank
    int my_rank, n_proc;
    int shm_my_rank, shm_n_proc;
    MPI_Comm shm_comm;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_size(shm_comm, &shm_n_proc);
    MPI_Comm_rank(shm_comm, &shm_my_rank);

    if (send_rank < 0 || send_rank >= n_proc || recv_rank < 0 || recv_rank >= n_proc)
    {
        send_rank = 1;
        recv_rank = 0;
    }

    // Get CUDA device status and set target CUDA device
    cuda_dev_state_p self_dev_state;
    cuda_init_dev_state(&self_dev_state);
    cuda_set_dev_id(self_dev_state, shm_my_rank % self_dev_state->n_dev);
    printf(
        "MPI rank %2d: host hash = %10u, shm_rank = %2d, bind to GPU %2d\n",
        my_rank, self_dev_state->host_hash, shm_my_rank, self_dev_state->dev_id
    );
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Set up signal array
    int n_msg = 16;
    size_t arr_bytes = sizeof(int) * n_msg;
    int *host_arr, *dev_arr;
    host_arr = (int *) malloc(arr_bytes);
    memset(host_arr, 0, arr_bytes);
    if (my_rank == send_rank) 
        for (int i = 0; i < n_msg; i++) host_arr[i] = 1;
    cuda_malloc_dev((void **) &dev_arr, arr_bytes);
    cuda_memcpy_h2d(host_arr, dev_arr, arr_bytes);
    cuda_device_sync();
    MPI_Barrier(MPI_COMM_WORLD);

    // Set up MPI window for DMA
    MPI_Info mpi_info;
    MPI_Win  mpi_win;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(dev_arr, arr_bytes, sizeof(int), mpi_info, MPI_COMM_WORLD, &mpi_win);
    MPI_Info_free(&mpi_info);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock_all(0, mpi_win);
    if (my_rank == send_rank)
    {
        sleep(1);  // Make sure that recv_rank kernel has launched and busy waiting
        MPI_Put(dev_arr, n_msg, MPI_INT, recv_rank, 0, n_msg, MPI_INT, mpi_win);
        printf("Rank %d MPI_Put returned\n", my_rank); fflush(stdout);
        MPI_Win_flush(recv_rank, mpi_win);
        printf("Rank %d MPI_Win_flush returned\n", my_rank); fflush(stdout);
    }
    //MPI_Barrier(MPI_COMM_WORLD);  // Without this barrier test kernel won't give the flag == 16 correct result
    if (my_rank == recv_rank)
    {
        launch_test_kernel(n_msg, dev_arr);
        printf("Rank %d launch_test_kernel returned\n", my_rank); fflush(stdout);

        launch_test_kernel(n_msg, dev_arr);
        printf("Rank %d launch_test_kernel returned\n", my_rank); fflush(stdout);
    }
    printf("Rank %d ready to call MPI_Win_unlock_all\n", my_rank); fflush(stdout);
    MPI_Win_unlock_all(mpi_win);
    printf("Rank %d MPI_Win_unlock_all returned\n", my_rank); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Clean up and exit
    cuda_free_dev(dev_arr);
    free(host_arr);
    cuda_free_dev_state(&self_dev_state);
    MPI_Win_free(&mpi_win);
    MPI_Comm_free(&shm_comm);
    MPI_Finalize();
    return 0;
}