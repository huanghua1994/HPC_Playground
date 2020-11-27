#ifndef __MPI_DEV_MEM_H__
#define __MPI_DEV_MEM_H__

#ifdef __cplusplus
extern "C" {
#endif

void MPI_proxy_init(int *argc, char ***argv);

void MPI_proxy_comm_size(void *mpi_comm, int *size);

void MPI_proxy_comm_rank(void *mpi_comm, int *rank);

void MPI_proxy_get_processor_name(char *name, int *res_len);

void MPI_proxy_barrier(void *mpi_comm);

void MPI_proxy_finalize();

int  MPI_proxy_get_local_rank_env();

void MPI_test_dev_mem_recv(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec1);

void MPI_test_dev_mem_put(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec2);

#ifdef __cplusplus
}
#endif

#endif
