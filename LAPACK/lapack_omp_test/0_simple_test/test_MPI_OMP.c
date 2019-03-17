
#define _BSD_SOURCE
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#include <unistd.h>
#include <time.h> 

#include <omp.h>


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // Initialize MPI
    
    int rank, nproc, i, N, irepeat;
    double t1, t2;
    time_t curr_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank );
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    printf("MPI rank %d / %d launched\n", rank + 1, nproc);
    
    N = 1000000000; // 1e9
    
    double *A, sum;
    if (rank == 0) 
    {
        A = (double *) malloc(N * sizeof(double));
        if (A == NULL) 
        {
            printf("Cannot allocate memory!\n");
            exit(1);
        }
        
        for (i = 0; i < N; i++) A[i] = 1.0;
    }

    sum = 0.0;
    
    // check env for number of threads to use
    int my_nthreads = 1;
    char *env_ntheads = getenv("NTHREADS"); 
    if (env_ntheads != NULL) my_nthreads = atoi(env_ntheads);
    if (my_nthreads < 1) my_nthreads = 1;
    
    curr_t = time(NULL);
    printf("Rank %d before barrier at: %s\n", rank, ctime(&curr_t));
    
    MPI_Request req;
    MPI_Barrier(MPI_COMM_WORLD);
    
    curr_t = time(NULL);
    printf("Rank %d just past barrier at: %s\n", rank, ctime(&curr_t));
    
    t1 = MPI_Wtime();
    // start finding sum of array A on rank 0
    if (rank == 0) 
    {
        omp_set_dynamic(0); // Explicitly disable dynamic teams
        omp_set_num_threads(my_nthreads);
        
        #pragma omp parallel
        {
            printf("MPI Rank 0, thread %d launched\n", omp_get_thread_num());

            #pragma omp for reduction(+:sum) 
            for (i = 0; i < N; i++) {
               sum += A[i];
            }
            
            #pragma omp barrier
        }
        
        printf("Calc done, MPI 0 thread %d still exists\n", omp_get_thread_num());
        
        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Ibarrier(MPI_COMM_WORLD, &req);
    } else {
        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Ibarrier(MPI_COMM_WORLD, &req);
    }
    
    // every one waits here, check every 10 ms to see if it's completed
    int flag;
    MPI_Status status;
	MPI_Test(&req, &flag, &status);
	while (!flag)
	{
	    /* Do some work ... */
	    usleep(10000); // in micro-seconds, this function is not preferred
	    
	    /* nanosleep is preferred, but not as handy */
		//struct timespec ts;
		//int milliseconds = 10;
		//ts.tv_sec = milliseconds / 1000;
		//ts.tv_nsec = (milliseconds % 1000) * 1000000;
		//nanosleep(&ts, NULL);
	    
	    MPI_Test(&req, &flag, &status);
	}

    t2 = MPI_Wtime();
    
    curr_t = time(NULL);
    printf("Rank %d ready to exit at: %s\n", rank, ctime(&curr_t));
    
    if (rank == 0) 
    {
        printf("Finding sum using %d threads took: %.3f ms, sum = %.6e\n", my_nthreads, (t2-t1)*1e3, sum);
        free(A);
    }
    
    MPI_Finalize();
    return 0;
}
