#ifndef __MPI_DEV_MEM_H__
#define __MPI_DEV_MEM_H__

#ifdef __cplusplus
extern "C" {
#endif

void MPI_Init_wrapper(int *argc, char ***argv);

void MPI_Comm_world_size_rank(int *n_proc, int *my_rank);

void MPI_Finalize_wrapper();

void MPI_test_dev_mem_recv(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec1);

void MPI_test_dev_mem_put(const int n_proc, const int my_rank, const int vec_len, int *d_vec0, int *d_vec2);

#ifdef __cplusplus
}
#endif

#endif
