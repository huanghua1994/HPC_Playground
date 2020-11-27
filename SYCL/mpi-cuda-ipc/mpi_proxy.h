#ifndef __MPI_PROXY_H__
#define __MPI_PROXY_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get current process MPI intra-node / global rank, number of intra-node / global 
// processes from environment variables (before initializing MPI)
int  MPI_proxy_get_local_rank_env();
int  MPI_proxy_get_local_size_env();
int  MPI_proxy_get_global_rank_env();
int  MPI_proxy_get_global_size_env();

void MPI_proxy_init(int *argc, char ***argv);
void MPI_proxy_init_thread_multiple(int *argc, char ***argv);
void MPI_proxy_get_processor_name(char *name, int *res_len);
void MPI_proxy_finalize();

void MPI_proxy_comm_size(void *mpi_comm, int *size);
void MPI_proxy_comm_rank(void *mpi_comm, int *rank);

void MPI_proxy_barrier(void *mpi_comm);
void MPI_proxy_barrier_sleep(void *mpi_comm, int sleep_us);

void MPI_proxy_comm_split_shm(void *mpi_comm, int key, void **shm_comm);
void MPI_proxy_comm_free(void **mpi_comm);

void MPI_proxy_sendrecv(
    void *send_buf, int send_cnt, void *mpi_send_type, int dst, int send_tag, 
    void *recv_buf, int recv_cnt, void *mpi_recv_type, int src, int recv_tag, void *mpi_comm
);

void MPI_proxy_allreduce(
    void *send_buf, void *recv_buf, int cnt, 
    void *mpi_dtype, void *mpi_op, void *mpi_comm
);

void MPI_proxy_allgather(
    void *send_buf, int send_cnt, void *mpi_send_type,
    void *recv_buf, int recv_cnt, void *mpi_recv_type, void *mpi_comm
);

void MPI_proxy_comm_group(void *mpi_comm, void **mpi_group);
void MPI_proxy_group_incl(void *mpi_group, int n, int *ranks, void **mpi_newgroup);

void MPI_proxy_win_create(
    void *base, size_t size, int disp_unit, void *mpi_info,
    void *mpi_comm, void **mpi_win
);
void MPI_proxy_win_free(void **mpi_win);

void MPI_proxy_win_lock_all(int assert, void *mpi_win);
void MPI_proxy_win_unlock_all(void *mpi_win);

void MPI_proxy_dtype_char(void **mpi_char);
void MPI_proxy_dtype_int(void **mpi_int);
void MPI_proxy_dtype_float(void **mpi_float);
void MPI_proxy_dtype_double(void **mpi_double);

void MPI_proxy_op_sum(void **mpi_sum);
void MPI_proxy_op_min(void **mpi_min);
void MPI_proxy_op_max(void **mpi_max);

#ifdef __cplusplus
}
#endif

#endif
