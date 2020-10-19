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
    cuda_dev_state_p self_dev_state;
    cuda_init_dev_state(&self_dev_state);
    cuda_set_dev_id(self_dev_state, my_local_rank % self_dev_state->n_dev);

    // If we are using MPICH + YAKSA, tell YAKSA to do lazy initialization
    setenv("YAKSA_LAZY_INIT_DEVICE", "1", 1);

    MPI_Init(&argc, &argv);

    int my_rank, n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int send_rank = 1, recv_rank = 0, tag = 1924;
    int use_cuda_p2p = 0, use_mpi_rma = 1, use_mpi_p2p = 0, use_mpi_ddt = 0;

    // When using Mellanox IB: 
    // 1. OpenMPI 4.0.5      : MPI send/recv & MPI DDT not working, error comes from UCX
    // 2. OpenMPI 3.1.6      : MPI send/recv & MPI DDT working
    // 3. MVAPICH2 2.3.{2,4} : MPI send/recv working, MPI DDT not working
    char *mv_use_cuda_p = getenv("MV2_USE_CUDA");
    if (mv_use_cuda_p != NULL) use_mpi_ddt = 0;

    if (argc >= 2) send_rank = atoi(argv[1]);
    if (argc >= 3) recv_rank = atoi(argv[2]);

    if (send_rank < 0 || send_rank >= n_proc || recv_rank < 0 || recv_rank >= n_proc)
    {
        send_rank = 1;
        recv_rank = 0;
    }

    // Gather all ranks' CUDA device status to build the topology map
    // using host_hash, pcie_dev_id, pcie_bus_id, and pcie_domain_id. 
    // We don't need a full topology map here, just need to check if 
    // the send / recv rank can do GPUDirect P2P access.
    int ds_msize = sizeof(cuda_dev_state_t);
    if (my_rank == recv_rank || my_rank == send_rank)
    {
        int peer_rank = (my_rank == send_rank) ? recv_rank : send_rank;
        int peer_host_hash, peer_dev_id;
        MPI_Request req0, req1;
        MPI_Isend(&self_dev_state->host_hash, 1, MPI_INT, peer_rank, 0, MPI_COMM_WORLD, &req0);
        MPI_Isend(&self_dev_state->dev_id,    1, MPI_INT, peer_rank, 1, MPI_COMM_WORLD, &req1);
        MPI_Recv(&peer_host_hash, 1, MPI_INT, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&peer_dev_id,    1, MPI_INT, peer_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        use_cuda_p2p = cuda_check_dev_p2p(
            self_dev_state->host_hash, self_dev_state->dev_id,
            peer_host_hash, peer_dev_id
        );
        if (use_cuda_p2p == 1) use_mpi_rma = 0;
    }
    printf(
        "MPI rank %2d: host hash = %10u, local_rank = %2d, bind to GPU %2d\n",
        my_rank, self_dev_state->host_hash, my_local_rank, self_dev_state->dev_id
    );
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

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
    if (my_rank == recv_rank) 
    {
        if (use_cuda_p2p) printf("Use CUDA P2P access"); 
        if (use_mpi_rma)  printf("Use MPI RMA access"); 
        if (use_mpi_p2p)  printf("Use MPI P2P access"); 
        if (use_mpi_rma || use_mpi_p2p) 
        {
            if (use_mpi_ddt == 1) printf(" with MPI DDT");
            else printf(" with MPI original data type");
        }
        printf(" for rank %d --> %d\n", send_rank, recv_rank);
        fflush(stdout);
    }

    // Set up MPI window for DMA
    MPI_Info mpi_info;
    MPI_Win  mpi_win;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(dev_mat, mat_bytes, sizeof(int), mpi_info, MPI_COMM_WORLD, &mpi_win);
    MPI_Info_free(&mpi_info);

    // Set up an MPI derived data type for the target block
    int nrow_send = nrow - 3;
    int ncol_send = ncol - 2;
    MPI_Datatype ddt_int_block;
    MPI_Type_vector(nrow_send, ncol_send, ncol, MPI_INT, &ddt_int_block);
    MPI_Type_commit(&ddt_int_block);

    // Time to communicate!
    if (use_cuda_p2p)
    {
        int handle_bytes;
        void *mem_handle;  // Allocated by get_cuda_ipc_mem_handle()
        cuda_get_ipc_mem_handle(dev_mat, &handle_bytes, &mem_handle);
        if (my_rank == send_rank) MPI_Send(mem_handle, handle_bytes, MPI_CHAR, recv_rank, tag, MPI_COMM_WORLD);
        if (my_rank == recv_rank) 
        {
            int *peer_dptr;
            MPI_Recv(mem_handle, handle_bytes, MPI_CHAR, send_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cuda_open_ipc_mem_handle((void **) &peer_dptr, mem_handle);
            for (int i = 0; i < nrow_send; i++)
            {
                int *src_ptr = peer_dptr + i * ncol;
                int *dst_ptr = dev_mat   + i * ncol;
                cuda_memcpy_auto(src_ptr, dst_ptr, sizeof(int) * ncol_send);
            }
            cuda_device_sync();
            cuda_close_ipc_mem_handle((void *) peer_dptr);
        }
    } 
    if (use_mpi_rma)
    {
        if (my_rank == send_rank)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, recv_rank, 0, mpi_win);
            if (use_mpi_ddt == 1)
            {
                MPI_Put(dev_mat, 1, ddt_int_block, recv_rank, 0, 1, ddt_int_block, mpi_win);
            } else {
                for (int i = 0; i < nrow_send; i++)
                {
                    int *send_ptr = dev_mat + i * ncol;
                    MPI_Aint target_disp = i * ncol;
                    MPI_Put(send_ptr, ncol_send, MPI_INT, recv_rank, target_disp, ncol_send, MPI_INT, mpi_win);
                }
            }
            MPI_Win_unlock(recv_rank, mpi_win);
        }
    }
    if (use_mpi_p2p) 
    {
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

    // Clean up and exit
    cuda_free_dev(dev_mat);
    free(host_mat);
    cuda_free_dev_state(&self_dev_state);
    MPI_Type_free(&ddt_int_block);
    MPI_Win_free(&mpi_win);
    MPI_Finalize();
    return 0;
}