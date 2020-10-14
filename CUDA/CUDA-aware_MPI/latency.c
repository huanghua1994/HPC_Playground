#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <mpi.h>
#include <unistd.h>

#include "cuda_proxy.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // Set up MPI shared memory communicator to get shared memory rank
    int my_rank, n_proc;
    int shm_my_rank, shm_n_proc;
    MPI_Comm shm_comm;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_size(shm_comm, &shm_n_proc);
    MPI_Comm_rank(shm_comm, &shm_my_rank);

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

    // Set up test arrays
    int arr_size = 64;
    if (argc >= 2) arr_size = atoi(argv[1]);
    if (arr_size < 32) arr_size = 32;
    if (my_rank == 0) printf("arr_size = %d\n", arr_size);
    size_t arr_bytes = sizeof(int) * arr_size;
    int *host_arr = (int *) malloc(arr_bytes * 2);
    int *dev_arr;
    cuda_malloc_dev((void **) &dev_arr, arr_bytes * 2);

    // Set up groups for PSCW sync
    int n_test = 10, tag = 1924;
    int prev_rank = (my_rank > 0) ? (my_rank - 1) : (n_proc - 1);
    int next_rank = (my_rank + 1) % n_proc;
    MPI_Group world_group, src_group, dst_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 1, &prev_rank, &src_group);
    MPI_Group_incl(world_group, 1, &next_rank, &dst_group);
    double st, et;

    // Test CPU send-receive latency
    double cpu_sr_t = 0.0;
    for (int i = 0; i <= n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        MPI_Sendrecv(
            host_arr + 0 * arr_size, arr_size, MPI_INT, next_rank, tag,
            host_arr + 1 * arr_size, arr_size, MPI_INT, prev_rank, tag, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        et = MPI_Wtime();
        if (i > 0) cpu_sr_t += et - st;
    }
    cpu_sr_t /= (double) n_test;
    printf(
        "Rank %2d: MPI_Sendrecv to/from (%2d, %2d) on host avg time = %.3lf ms\n", 
        my_rank, next_rank, prev_rank, cpu_sr_t * 1000.0
    );
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Test CPU window put
    MPI_Win host_win;
    MPI_Win_create(host_arr, arr_bytes, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &host_win);
    double cpu_put_t = 0.0;
    for (int i = 0; i <= n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        MPI_Win_post(src_group, 0, host_win);
        MPI_Win_start(dst_group, 0, host_win);
        MPI_Put(host_arr, arr_size, MPI_INT, next_rank, arr_size, arr_size, MPI_INT, host_win);
        MPI_Win_complete(host_win);
        MPI_Win_wait(host_win);
        et = MPI_Wtime();
        if (i > 0) cpu_put_t += et - st;
    }
    MPI_Win_free(&host_win);
    cpu_put_t /= (double) n_test;
    printf(
        "Rank %2d: MPI window PSCW + put to (%2d) on host avg time = %.3lf ms\n", 
        my_rank, next_rank, cpu_put_t * 1000.0
    );
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Test GPU send-receive latency
    double gpu_sr_t = 0.0;
    for (int i = 0; i <= n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        MPI_Sendrecv(
            dev_arr + 0 * arr_size, arr_size, MPI_INT, next_rank, tag,
            dev_arr + 1 * arr_size, arr_size, MPI_INT, prev_rank, tag, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        et = MPI_Wtime();
        if (i > 0) gpu_sr_t += et - st;
    }
    gpu_sr_t /= (double) n_test;
    printf(
        "Rank %2d: MPI_Sendrecv to/from (%2d, %2d) on CUDA avg time = %.3lf ms\n", 
        my_rank, next_rank, prev_rank, gpu_sr_t * 1000.0
    );
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Test GPU window put
    MPI_Win cuda_win;
    MPI_Win_create(dev_arr, arr_bytes, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &cuda_win);
    double gpu_put_t = 0.0;
    for (int i = 0; i <= n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        MPI_Win_post(src_group, 0, cuda_win);
        MPI_Win_start(dst_group, 0, cuda_win);
        MPI_Put(dev_arr, arr_size, MPI_INT, next_rank, arr_size, arr_size, MPI_INT, cuda_win);
        MPI_Win_complete(cuda_win);
        MPI_Win_wait(cuda_win);
        et = MPI_Wtime();
        if (i > 0) gpu_put_t += et - st;
    }
    MPI_Win_free(&cuda_win);
    gpu_put_t /= (double) n_test;
    printf(
        "Rank %2d: MPI window PSCW + put to (%2d) on CUDA avg time = %.3lf ms\n", 
        my_rank, next_rank, gpu_put_t * 1000.0
    );
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    free(host_arr);
    cuda_free_dev(dev_arr);
    MPI_Finalize();
}