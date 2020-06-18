#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int nproc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int src[3], dst[3], weights[3];
    int send_cnt[3], send_displs[4], recv_cnt[3], recv_displs[4];
    src[0] = (nproc + rank - 3) % nproc;
    src[1] = (nproc + rank - 1) % nproc;
    src[2] = (nproc + rank - 2) % nproc;
    dst[0] = (nproc + rank + 2) % nproc;
    dst[1] = (nproc + rank + 1) % nproc;
    dst[2] = (nproc + rank + 3) % nproc;
    int send_size = rank * (dst[0] + dst[1] + dst[2]);
    int recv_size = rank * (src[0] + src[1] + src[2]);
    int *send_buf = (int*) malloc(sizeof(int) * send_size);
    int *recv_buf = (int*) malloc(sizeof(int) * recv_size);
    int cnt = 0;
    for (int i = 0; i < rank * dst[0]; i++) 
        send_buf[cnt + i] = 100 * rank + dst[0];
    cnt += rank * dst[0];
    for (int i = 0; i < rank * dst[1]; i++) 
        send_buf[cnt + i] = 100 * rank + dst[1];
    cnt += rank * dst[1];
    for (int i = 0; i < rank * dst[2]; i++) 
        send_buf[cnt + i] = 100 * rank + dst[2];
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for (int i = 0; i < 3; i++)
    {
        send_cnt[i] = rank * dst[i];
        recv_cnt[i] = rank * src[i];
        send_displs[i + 1] = send_displs[i] + send_cnt[i];
        recv_displs[i + 1] = recv_displs[i] + recv_cnt[i];
    }

    MPI_Comm neighbor_comm;
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, 3, src, MPI_UNWEIGHTED, 3, dst, MPI_UNWEIGHTED, info, 0, &neighbor_comm);
    int src1[3], dst1[3];
    // src1/dst1 should be the same as src/dst, just to check here
    MPI_Dist_graph_neighbors(neighbor_comm, 3, src1, weights, 3, dst1, weights);
    printf("Rank %d get src = %d, %d, %d, dst = %d, %d, %d\n", rank, src1[0], src1[1], src1[2], dst1[0], dst1[1], dst1[2]);

    MPI_Neighbor_alltoallv(
        send_buf, send_cnt, send_displs, MPI_INT, 
        recv_buf, recv_cnt, recv_displs, MPI_INT, neighbor_comm
    );

    for (int i = 0; i < nproc; i++)
    {
        if (i == rank)
        {
            printf("Rank %d received:\n", rank);
            for (int j = 0; j < rank * src[0]; j++) 
                printf("%d ", recv_buf[recv_displs[0] + j]);
            printf("\n");
            for (int j = 0; j < rank * src[1]; j++) 
                printf("%d ", recv_buf[recv_displs[1] + j]);
            printf("\n");
            for (int j = 0; j < rank * src[2]; j++) 
                printf("%d ", recv_buf[recv_displs[2] + j]);        
            printf("\n");
            fflush(stdout);
        }
        usleep(1000 * 50);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(send_buf);
    free(recv_buf);
    MPI_Finalize();
    return 0;
}