#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>

void MPI_proxy_init(int *argc, char ***argv)
{
    MPI_Init(argc, argv);
}

void MPI_proxy_comm_size(void *mpi_comm, int *size)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Comm_size(comm, size);
}

void MPI_proxy_comm_rank(void *mpi_comm, int *rank)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Comm_rank(comm, rank);
}


void MPI_proxy_get_processor_name(char *name, int *res_len)
{
    MPI_Get_processor_name(name, res_len);
}

void MPI_proxy_barrier(void *mpi_comm)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Barrier(comm);
}

void MPI_proxy_finalize() 
{ 
    MPI_Finalize(); 
}

int  MPI_proxy_get_local_rank_env()
{
    int local_rank = -1;
    char *env_p;

    // MPICH
    env_p = getenv("MPI_LOCALRANKID");
    if (env_p != NULL) return atoi(env_p);

    // MVAPICH2
    env_p = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (env_p != NULL) return atoi(env_p);

    // OpenMPI
    env_p = getenv("OMPI_COMM_WORLD_NODE_RANK");
    if (env_p != NULL) return atoi(env_p);

    // SLURM or PBS/Torque
    env_p = getenv("SLURM_LOCALID");
    if (env_p != NULL) return atoi(env_p);

    env_p = getenv("PBS_O_VNODENUM");
    if (env_p != NULL) return atoi(env_p);

    return local_rank;
}

void MPI_test_dev_mem_recv(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec1)
{
    int next_rank = (my_rank + 1) % n_proc;
    int prev_rank = (my_rank == 0) ? (n_proc - 1) : my_rank - 1;
    MPI_Request req;
    MPI_Isend(d_vec0, vec_len, MPI_INT, next_rank, 42, MPI_COMM_WORLD, &req);
    MPI_Recv(d_vec1, vec_len, MPI_INT, prev_rank, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
}

void MPI_test_dev_mem_put(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec2)
{
    int next_rank = (my_rank + 1) % n_proc;
    int prev_rank = (my_rank == 0) ? (n_proc - 1) : my_rank - 1;
    MPI_Win mpi_win;
    MPI_Win_create(d_vec2, sizeof(int) * vec_len, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &mpi_win);
    MPI_Win_lock(MPI_LOCK_SHARED, prev_rank, 0, mpi_win);
    MPI_Put(
        d_vec0, vec_len, MPI_INT, prev_rank, 
        0,      vec_len, MPI_INT, mpi_win
    );
    MPI_Win_unlock(prev_rank, mpi_win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&mpi_win);
    usleep(1 * 1000);
}
