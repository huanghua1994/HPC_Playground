#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int glb_nproc, glb_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &glb_nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &glb_rank);

    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(proc_name, &name_len);

    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        glb_rank, MPI_INFO_NULL, &shm_comm);

    int shm_nproc, shm_rank;
    MPI_Comm_size(shm_comm, &shm_nproc);
    MPI_Comm_rank(shm_comm, &shm_rank);

    for (int i = 0; i < glb_nproc; i++)
    {
        if (i == glb_rank)
        {
            printf(
                "Global rank %5d / %5d @ host %s, shmem rank %3d / %3d\n",
                glb_rank, glb_nproc, proc_name, shm_rank, shm_nproc
            );
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
