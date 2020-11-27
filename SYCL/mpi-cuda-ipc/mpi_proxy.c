#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>
#include "mpi_proxy.h"

static const MPI_Datatype dtype_char   = MPI_CHAR;
static const MPI_Datatype dtype_int    = MPI_INT;
static const MPI_Datatype dtype_float  = MPI_FLOAT;
static const MPI_Datatype dtype_double = MPI_DOUBLE;

static const MPI_Op op_sum = MPI_SUM;
static const MPI_Op op_min = MPI_MIN;
static const MPI_Op op_max = MPI_MAX;

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

int  MPI_proxy_get_local_size_env()
{
    int local_size = -1;
    char *env_p;

    // MPICH
    env_p = getenv("MPI_LOCALNRANKS");
    if (env_p != NULL) return atoi(env_p);

    // MVAPICH2
    env_p = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
    if (env_p != NULL) return atoi(env_p);

    // OpenMPI
    env_p = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
    if (env_p != NULL) return atoi(env_p);

    return local_size;
}

int  MPI_proxy_get_global_rank_env()
{
    int local_rank = -1;
    char *env_p;

    // MPICH
    // ???

    // MVAPICH2
    env_p = getenv("MV2_COMM_WORLD_RANK");
    if (env_p != NULL) return atoi(env_p);

    // OpenMPI
    env_p = getenv("OMPI_COMM_WORLD_RANK");
    if (env_p != NULL) return atoi(env_p);

    // SLURM or PBS/Torque
    env_p = getenv("SLURM_PROCID");
    if (env_p != NULL) return atoi(env_p);

    env_p = getenv("PBS_O_TASKNUM");
    if (env_p != NULL) return (atoi(env_p) - 1);

    return local_rank;
}

int  MPI_proxy_get_global_size_env()
{
    int local_size = -1;
    char *env_p;

    // MPICH
    // ???

    // MVAPICH2
    env_p = getenv("MV2_COMM_WORLD_SIZE");
    if (env_p != NULL) return atoi(env_p);

    // OpenMPI
    env_p = getenv("OMPI_COMM_WORLD_SIZE");
    if (env_p != NULL) return atoi(env_p);

    // SLURM or PBS/Torque
    env_p = getenv("SLURM_NTASKS");
    if (env_p != NULL) return atoi(env_p);

    env_p = getenv("PBS_NP");
    if (env_p != NULL) return atoi(env_p);

    return local_size;
}

void MPI_proxy_init(int *argc, char ***argv)
{
    MPI_Init(argc, argv);
}

void MPI_proxy_init_thread_multiple(int *argc, char ***argv)
{
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
        printf("[WARNING] MPI_Init_thread required %d, provided %d\n", MPI_THREAD_MULTIPLE, provided);
}

void MPI_proxy_get_processor_name(char *name, int *res_len)
{
    MPI_Get_processor_name(name, res_len);
}

void MPI_proxy_finalize() 
{ 
    MPI_Finalize(); 
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

void MPI_proxy_barrier(void *mpi_comm)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Barrier(comm);
}

void MPI_proxy_barrier_sleep(void *mpi_comm, int sleep_us)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Request req;
    MPI_Ibarrier(comm, &req);
    int flag = 0;
    while (flag == 0)
    {
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        usleep(sleep_us);
    }
}

void MPI_proxy_comm_split_shm(void *mpi_comm, int key, void **shm_comm)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Comm *shm_comm_ptr = (MPI_Comm *) malloc(sizeof(MPI_Comm));
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL, shm_comm_ptr);
    *shm_comm = shm_comm_ptr;
}

void MPI_proxy_comm_free(void **mpi_comm)
{
    MPI_Comm *comm_ptr = *mpi_comm;
    MPI_Comm_free(comm_ptr);
    free(comm_ptr);
}

void MPI_proxy_sendrecv(
    void *send_buf, int send_cnt, void *mpi_send_type, int dst, int send_tag, 
    void *recv_buf, int recv_cnt, void *mpi_recv_type, int src, int recv_tag, void *mpi_comm
)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Datatype send_type = *((MPI_Datatype *) mpi_send_type);
    MPI_Datatype recv_type = *((MPI_Datatype *) mpi_recv_type);
    MPI_Sendrecv(
        send_buf, send_cnt, send_type, dst, send_tag,
        recv_buf, recv_cnt, recv_type, src, recv_tag, comm, MPI_STATUS_IGNORE
    );
}

void MPI_proxy_allreduce(
    void *send_buf, void *recv_buf, int cnt, 
    void *mpi_dtype, void *mpi_op, void *mpi_comm
)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Datatype datatype = *((MPI_Datatype *) mpi_dtype);
    MPI_Op op = *((MPI_Op *) mpi_op);
    MPI_Allreduce(send_buf, recv_buf, cnt, datatype, op, comm);
}

void MPI_proxy_allgather(
    void *send_buf, int send_cnt, void *mpi_send_type,
    void *recv_buf, int recv_cnt, void *mpi_recv_type, void *mpi_comm
)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Datatype send_type = *((MPI_Datatype *) mpi_send_type);
    MPI_Datatype recv_type = *((MPI_Datatype *) mpi_recv_type);
    MPI_Allgather(
        send_buf, send_cnt, send_type,
        recv_buf, recv_cnt, recv_type, comm
    );
}

void MPI_proxy_comm_group(void *mpi_comm, void **mpi_group)
{
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Group *group_ptr = (MPI_Group *) malloc(sizeof(MPI_Group));
    MPI_Comm_group(comm, group_ptr);
    *mpi_group = group_ptr;
}

void MPI_proxy_group_incl(void *mpi_group, int n, int *ranks, void **mpi_newgroup)
{
    MPI_Group group = *((MPI_Group *) mpi_group);
    MPI_Group *newgroup_ptr = (MPI_Group *) malloc(sizeof(MPI_Group));
    MPI_Group_incl(group, n, ranks, newgroup_ptr);
    *mpi_newgroup = newgroup_ptr;
}

void MPI_proxy_win_create(
    void *base, size_t size, int disp_unit, void *mpi_info,
    void *mpi_comm, void **mpi_win
)
{
    MPI_Info info = (mpi_info == NULL) ? MPI_INFO_NULL  : *((MPI_Info *) mpi_info);
    MPI_Comm comm = (mpi_comm == NULL) ? MPI_COMM_WORLD : *((MPI_Comm *) mpi_comm);
    MPI_Win *win_ptr = (MPI_Win *) malloc(sizeof(MPI_Win));
    MPI_Win_create(base, size, disp_unit, info, comm, win_ptr);
    *mpi_win = win_ptr;
}

void MPI_proxy_win_free(void **mpi_win)
{
    MPI_Win *win_ptr = (MPI_Win *) (*mpi_win);
    MPI_Win_free(win_ptr);
    free(win_ptr);
}

void MPI_proxy_win_lock_all(int assert, void *mpi_win)
{
    MPI_Win win = *((MPI_Win *) mpi_win);
    MPI_Win_lock_all(assert, win);
}

void MPI_proxy_win_unlock_all(void *mpi_win)
{
    MPI_Win win = *((MPI_Win *) mpi_win);
    MPI_Win_unlock_all(win);
}

void MPI_proxy_dtype_char(void **mpi_char)
{
    MPI_Datatype *dtype = (MPI_Datatype *) malloc(sizeof(MPI_Datatype));
    memcpy(dtype, &dtype_char, sizeof(MPI_Datatype));
    *mpi_char = dtype;
}

void MPI_proxy_dtype_int(void **mpi_int)
{
    MPI_Datatype *dtype = (MPI_Datatype *) malloc(sizeof(MPI_Datatype));
    memcpy(dtype, &dtype_int, sizeof(MPI_Datatype));
    *mpi_int = dtype;
}

void MPI_proxy_dtype_float(void **mpi_float)
{
    MPI_Datatype *dtype = (MPI_Datatype *) malloc(sizeof(MPI_Datatype));
    memcpy(dtype, &dtype_float, sizeof(MPI_Datatype));
    *mpi_float = dtype;
}

void MPI_proxy_dtype_double(void **mpi_double)
{
    MPI_Datatype *dtype = (MPI_Datatype *) malloc(sizeof(MPI_Datatype));
    memcpy(dtype, &dtype_double, sizeof(MPI_Datatype));
    *mpi_double = dtype;
}

void MPI_proxy_op_sum(void **mpi_sum)
{
    MPI_Op *op = (MPI_Op *) malloc(sizeof(MPI_Op));
    memcpy(op, &op_sum, sizeof(MPI_Op));
    *mpi_sum = op;
}

void MPI_proxy_op_max(void **mpi_max)
{
    MPI_Op *op = (MPI_Op *) malloc(sizeof(MPI_Op));
    memcpy(op, &op_max, sizeof(MPI_Op));
    *mpi_max = op;
}

void MPI_proxy_op_min(void **mpi_min)
{
    MPI_Op *op = (MPI_Op *) malloc(sizeof(MPI_Op));
    memcpy(op, &op_min, sizeof(MPI_Op));
    *mpi_min = op;
}