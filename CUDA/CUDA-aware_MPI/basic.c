#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <mpi.h>

#include "cuda_proxy.h"

int main(int argc, char **argv)
{
    // Get CUDA device status and set target CUDA device
    int my_local_rank = get_mpi_local_rank_env();
    int n_local_proc  = get_mpi_local_size_env(); 
    cuda_dev_state_p self_dev_state;
    cuda_init_dev_state(&self_dev_state);
    cuda_set_dev_id(self_dev_state, my_local_rank % self_dev_state->n_dev);

    MPI_Init(&argc, &argv);

    int my_rank, n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    printf(
        "MPI rank %2d/%2d, local rank %2d/%2d, use GPU %d\n",
        my_rank, n_proc, my_local_rank, n_local_proc, self_dev_state->dev_id
    );

    int send_rank = 1, recv_rank = 0;
    if (argc >= 2) send_rank = atoi(argv[1]);
    if (argc >= 3) recv_rank = atoi(argv[2]);
    if (send_rank < 0 || send_rank >= n_proc || recv_rank < 0 || recv_rank >= n_proc)
    {
        send_rank = 1;
        recv_rank = 0;
    }

    // Set up the matrix
    int nrow = 16, ncol = 16;
    size_t mat_bytes = sizeof(int) * nrow * ncol;
    int *host_mat, *dev_mat;
    host_mat = (int *) malloc(mat_bytes);
    cuda_malloc_dev((void **) &dev_mat, mat_bytes);
    memset(host_mat, 0, mat_bytes);
    if (my_rank == send_rank)
    {
        for (int i = 0; i < nrow; i++)
            for (int j = 0; j < ncol; j++) 
                host_mat[i * ncol + j] = my_rank * 100 + i * ncol + j;
    }
    cuda_memcpy_h2d(host_mat, dev_mat, mat_bytes);
    cuda_device_sync();
    MPI_Barrier(MPI_COMM_WORLD);

    // Set up an MPI derived data type for the target block
    int nrow_send = nrow - 3;
    int ncol_send = ncol - 2;
    MPI_Datatype ddt_int_block;
    MPI_Type_vector(nrow_send, ncol_send, ncol, MPI_INT, &ddt_int_block);
    MPI_Type_commit(&ddt_int_block);

    // GPUDirect communication
    int tag = 42;
    int use_mpi_ddt = 0;
    if (use_mpi_ddt == 1) 
    {
        if (my_rank == send_rank) MPI_Send(dev_mat, 1, ddt_int_block, recv_rank, tag, MPI_COMM_WORLD);
        if (my_rank == recv_rank) MPI_Recv(dev_mat, 1, ddt_int_block, send_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        if (my_rank == send_rank)
        {
            for (int i = 0; i < nrow_send; i++)
                MPI_Send(dev_mat + i * ncol, ncol_send, MPI_INT, recv_rank, tag + i, MPI_COMM_WORLD);
        }
        if (my_rank == recv_rank)
        {
            for (int i = 0; i < nrow_send; i++)
                MPI_Recv(dev_mat + i * ncol, ncol_send, MPI_INT, send_rank, tag + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == recv_rank) printf("Communication done\n");

    // Print the received matrix to check the correctness
    if (my_rank == recv_rank)
    {
        cuda_memcpy_d2h(dev_mat, host_mat, mat_bytes);
        cuda_device_sync();
        printf("Rank %d, received matrix:\n", my_rank);
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++) printf("%3d ", host_mat[i * ncol + j]);
            printf("\n");
        }
        printf("\n");
    }

    cuda_free_dev(dev_mat);
    free(host_mat);
    MPI_Type_free(&ddt_int_block);
    MPI_Finalize();
}