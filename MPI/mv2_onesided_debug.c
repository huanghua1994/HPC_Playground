// Compiler     : ICC 19.0.3 & 19.0.5
// MPI Library  : MVAPICH2 2.3.2 & 2.3.3
// OS           : CentOS 7.6 x86_64, kernel: 3.10.0-957.12.1.el7.x86_64
// NIC          : Mellanox ConnectX-5, MLNX_OFED_LINUX-4.5-1.0.1.0
// Compile      : mpicc -O0 -g -std=gnu99 onesided_debug.c -o onesided_debug.exe
// Run          : SHM_QUERY=1 mpirun -np 8 ./onesided_debug.exe  <-- bug triggered
//                SHM_QUERY=0 mpirun -np 8 ./onesided_debug.exe  <-- bug not triggered
// 
// MVAPICH2 developer's reply: 
// > As you indicated, your program is creating multiple shared-memory windows and, 
// > most importantly, multiple communicators (through MPI_Comm_dup). Such duplication 
// > will eventually hit the limit that runtime allows to allocate resources for 
// > shared-memory. Once we hit the limit, MVAPICH2 runtime would not create more 
// > shared-memory windows, that is why you observe a NULL pointer for `shm_base_addrs`.
// > Please try to set the environment variable "MV2_SHMEM_COLL_NUM_COMM=64" and see if it works.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

struct mpi_global_vec
{
    MPI_Comm mpi_comm, shm_comm;
    MPI_Win  mpi_win,  shm_win; 
    MPI_Datatype datatype;
    size_t unit_size;
    int    my_rank, comm_size;
    int    shm_rank, shm_size;
    int    *displs;
    void   *vec_block;
    int    *shm_global_ranks;
    void   **shm_vec_blocks;
};
typedef struct mpi_global_vec  mpi_global_vec_s;
typedef struct mpi_global_vec* mpi_global_vec_t;

void mpi_global_vec_create(
    mpi_global_vec_t *glbvec_, MPI_Comm comm, MPI_Datatype datatype,
    size_t unit_size, int *displs
)
{
    mpi_global_vec_t glbvec = (mpi_global_vec_t) malloc(sizeof(mpi_global_vec_s));
    
    // Copy vector and displacement array
    MPI_Comm_dup (comm, &glbvec->mpi_comm);
    MPI_Comm_size(comm, &glbvec->comm_size);
    MPI_Comm_rank(comm, &glbvec->my_rank);
    glbvec->unit_size = unit_size;
    glbvec->datatype  = datatype;
    int my_rank   = glbvec->my_rank;
    int comm_size = glbvec->comm_size;
    size_t displs_msize = sizeof(int) * (comm_size + 1);
    glbvec->displs = (int*) malloc(displs_msize);
    memcpy(glbvec->displs, displs, displs_msize);
    int my_blksize = displs[my_rank + 1] - displs[my_rank];
    size_t my_blk_msize = unit_size * my_blksize;
    
    // Allocate shared memory and its MPI window
    // (1) Split communicator to get shared memory communicator
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &glbvec->shm_comm);
    MPI_Comm_rank(glbvec->shm_comm, &glbvec->shm_rank);
    MPI_Comm_size(glbvec->shm_comm, &glbvec->shm_size);
    glbvec->shm_global_ranks = (int*) malloc(sizeof(int) * glbvec->shm_size);
    MPI_Allgather(&my_rank, 1, MPI_INT, glbvec->shm_global_ranks, 1, MPI_INT, glbvec->shm_comm);
    // (2) Allocate shared memory
    MPI_Info shm_info;
    MPI_Info_create(&shm_info);
    MPI_Info_set(shm_info, "alloc_shared_noncontig", "true");
    MPI_Win_allocate_shared(
        my_blk_msize, unit_size, shm_info, glbvec->shm_comm, 
        &glbvec->vec_block, &glbvec->shm_win
    );
    MPI_Info_free(&shm_info);
    // (3) Get pointers of all processes in the shared memory communicator
    MPI_Aint _size;
    int _disp;
    glbvec->shm_vec_blocks = (void**) malloc(sizeof(void*) * glbvec->shm_size);
    int shm_query = 1;
    char *shm_query_p = getenv("SHM_QUERY");
    if (shm_query_p != NULL) 
    {
        shm_query = atoi(shm_query_p);
        if (shm_query < 0 || shm_query > 1) shm_query = 0;
    }
    if (my_rank == 0) printf("[INFO] MPI_Win_shared_query ? %d\n", shm_query);
    if (shm_query == 1)
    {
        for (int i = 0; i < glbvec->shm_size; i++)
            MPI_Win_shared_query(glbvec->shm_win, i, &_size, &_disp, &glbvec->shm_vec_blocks[i]);
    } else {
        for (int i = 0; i < glbvec->shm_size; i++)
            glbvec->shm_vec_blocks[i] = NULL;
    }
    
    // Bind local vector block to global MPI window
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(glbvec->vec_block, my_blk_msize, unit_size, mpi_info, glbvec->mpi_comm, &glbvec->mpi_win);
    MPI_Info_free(&mpi_info);
    
    *glbvec_ = glbvec;
}

void mpi_global_vec_free(mpi_global_vec_t glbvec)
{
    if (glbvec == NULL) return;
    
    MPI_Win_free (&glbvec->mpi_win);
    MPI_Win_free (&glbvec->shm_win); 
    MPI_Comm_free(&glbvec->mpi_comm);
    MPI_Comm_free(&glbvec->shm_comm);
    
    free(glbvec->displs);
    free(glbvec->shm_global_ranks);
    free(glbvec->shm_vec_blocks);
    
    free(glbvec);
}

int main(int argc, char **argv)
{
    int my_rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    const int n_glbvec = 16;
    const int nelem = 1000;
    mpi_global_vec_t *glbvecs = (mpi_global_vec_t*) malloc(sizeof(mpi_global_vec_t) * n_glbvec);
    int *displs = (int*) malloc(sizeof(int) * (nproc + 1));
    for (int i = 0; i <= nproc; i++) displs[i] = nelem * i;
    for (int i = 0; i < n_glbvec; i++)
    {
        //printf("Rank %2d ready to create mpi_global_vec %d (%d in total)\n", my_rank, i, n_glbvec);
        mpi_global_vec_create(&glbvecs[i], MPI_COMM_WORLD, MPI_DOUBLE, sizeof(int), displs);
        //printf("Rank %2d mpi_global_vec_create %d done\n", my_rank, i);
        int *vec_block = (int*) glbvecs[i]->vec_block;
        for (int j = 0; j < nelem; j++) vec_block[j] = 0;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    int *addend = (int*) malloc(sizeof(int) * nproc * nelem);
    for (int i = 0; i < nproc * nelem; i++) addend[i] = my_rank;
    for (int i = 0; i < n_glbvec; i++)
    {
        for (int r = 0; r < nproc; r++)
        {
            int *src_ptr = addend + displs[r];
            MPI_Win_lock(MPI_LOCK_SHARED, r, 0, glbvecs[i]->mpi_win);
            MPI_Accumulate(src_ptr, nelem, MPI_INT, r, 0, nelem, MPI_INT, MPI_SUM, glbvecs[i]->mpi_win);
            MPI_Win_unlock(r, glbvecs[i]->mpi_win);
        }
        //printf("Rank %2d accumulate to mpi_global_vec %d done\n", my_rank, i);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        int ref = nproc * (nproc - 1) / 2;
        int valid = 1, pos;
        int *vec_block = (int*) glbvecs[i]->vec_block;
        for (int j = 0; j < nelem; j++)
        {
            if (vec_block[j] != ref) 
            {
                valid = 0;
                pos = j;
            }
        }
        if (valid) printf("Rank %2d mpi_global_vec %d result correct\n", my_rank, i);
        else printf("Rank %2d mpi_global_vec %d result ERROR!! At location %d: %d <--> %d\n", my_rank, i, pos, vec_block[pos], ref);
    }
    
    for (int i = 0; i < n_glbvec; i++)
    {
        mpi_global_vec_free(glbvecs[i]);
        //printf("Rank %2d mpi_global_vec_free %d done\n", my_rank, i);
    }
    
    free(glbvecs);
    free(displs);
    
    MPI_Finalize();
    return 0;
}
