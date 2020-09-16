#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <mpi.h>

#include "driver.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int send_rank = 1, recv_rank = 0;

    int my_rank, n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int shm_my_rank, shm_n_proc;
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_size(shm_comm, &shm_n_proc);
    MPI_Comm_rank(shm_comm, &shm_my_rank);

    int n_gpu_dev = get_gpu_device_cnt();
    int my_gpu_id = shm_my_rank % n_gpu_dev;
    set_gpu_device(my_gpu_id);
    printf(
        "Rank %d / %d, shm rank %d / %d, bind to GPU %d / %d\n", 
        my_rank, n_proc, shm_my_rank, shm_n_proc, my_gpu_id, n_gpu_dev
    );

    int nrow = 16, ncol = 16;
    size_t mat_bytes = sizeof(int) * nrow * ncol;
    int *host_mat, *dev_mat;
    host_mat = (int *) malloc(mat_bytes);
    alloc_gpu_mem((void **) &dev_mat, mat_bytes);
    memset(host_mat, 0, mat_bytes);
    if (my_rank == send_rank)
    {
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++) 
            {
                int val = my_rank * 100 + i * ncol + j;
                host_mat[i * ncol + j] = val;
            }
        }
    }
    memcpy_h2d(host_mat, dev_mat, mat_bytes);
    sync_gpu_device();
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) printf("Ready to sendrecv\n");

    MPI_Info mpi_info;
    MPI_Win  mpi_win;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(dev_mat, mat_bytes, sizeof(int), mpi_info, MPI_COMM_WORLD, &mpi_win);
    MPI_Info_free(&mpi_info);

    int tag = 1924, use_rma = 1;
    int nrow_send = nrow - 3;
    int ncol_send = ncol - 2;
    MPI_Datatype ddt_int_block;
    MPI_Type_vector(nrow_send, ncol_send, ncol, MPI_INT, &ddt_int_block);
    MPI_Type_commit(&ddt_int_block);
    if (use_rma == 1)
    {
        if (my_rank == send_rank)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, recv_rank, 0, mpi_win);
            // On Cori's GPU node, MPI_Put with MPI DDT doesn't work with the 
            // following configurations:
            // (1) GCC 8.3 + UXC 1.8.1 + OpenMPI 4.0.3 (UCX does not support MPI_Put on CUDA)
            // (2) GCC 8.3 + MVAPICH2 2.3.2 (segmentation fault)
            // MPI_Put(dev_mat, 1, ddt_int_block, recv_rank, 0, 1, ddt_int_block, mpi_win);
            // However, the following loop works with GCC 8.3 + MVAPICH2 2.3.2
            for (int i = 0; i < nrow_send; i++)
            {
                int *send_addr = dev_mat + i * ncol;
                MPI_Aint target_disp = i * ncol;
                MPI_Put(send_addr, ncol_send, MPI_INT, recv_rank, target_disp, ncol_send, MPI_INT, mpi_win);
            }
            MPI_Win_unlock(recv_rank, mpi_win);
        }
    } else {
        if (my_rank == send_rank) MPI_Send(dev_mat, 1, ddt_int_block, recv_rank, tag, MPI_COMM_WORLD);
        if (my_rank == recv_rank) MPI_Recv(dev_mat, 1, ddt_int_block, send_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Type_free(&ddt_int_block);
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) printf("Sendrecv done\n");

    if (my_rank == recv_rank)
    {
        memcpy_d2h(dev_mat, host_mat, mat_bytes);
        sync_gpu_device();
        printf("Rank %d, received matrix:\n", my_rank);
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++) printf("%3d ", host_mat[i * ncol + j]);
            printf("\n");
        }
        printf("\n");
    }

    free_gpu_mem(dev_mat);
    free(host_mat);
    MPI_Win_free(&mpi_win);
    MPI_Comm_free(&shm_comm);
    MPI_Finalize();
    return 0;
}