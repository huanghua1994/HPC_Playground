#include <stdio.h>
#include <mkl.h>
#include <mpi.h>
#include <stdlib.h>

#include <unistd.h> // for 'usleep()'
#include <time.h> // for 'nanosleep()'

#include <omp.h>

#define min(x,y) ((x)<(y)?(x):(y))


int main(int argc, char **argv) {
    MPI_Init(&argc,&argv); // Initialize MPI
        
    int rank, nproc, i, j, k;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    printf("MPI rank %d / %d launched\n", rank + 1, nproc);

    double t1, t2;
    double *A, *B;
    int N;
    
    // define matrix size
    N = 5000;

    // define the matrix
    if (rank == 0) {
        srand(1);
        A = (double *)malloc(N*N*sizeof(double));
        B = (double *)calloc(N*N,sizeof(double));
        for (j = 0; j < N; j++) {
            for (i = 0; i < N; i++) {
                //A[j*N+i] = rand() / (double)RAND_MAX - 0.5;
                A[j*N+i] = 1.0;
            }
        }
    }
    
    // check env for number of threads to use
    int my_nthreads = 1;
    char *env_ntheads = getenv("NTHREADS"); 
    if (env_ntheads != NULL) 
        my_nthreads = atoi(env_ntheads);
    if (my_nthreads < 1) my_nthreads = 1;
    
    
    
    /**  Implementation one **/
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    // start finding B = A'*A on rank 0
    if (rank == 0) {
        omp_set_dynamic(0); // Explicitly disable dynamic teams
        omp_set_num_threads(1);
        omp_set_nested(1);
        mkl_set_dynamic(0);
        mkl_set_num_threads(64);
        #pragma omp parallel
        {   
            printf("Number of threads: %d\n", omp_get_num_threads());
            // create sym-pos-def matrix B = A' * A
            cblas_dgemm (CblasColMajor,CblasTrans,CblasNoTrans,
                         N,N,N,1,A,N,A,N,0,B,N);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    omp_set_num_threads(1);
    t2 = MPI_Wtime();
    
    */
    
    int save;
    
    /**  Implementation two **/
    MPI_Request req;
    //MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    // start finding B = A'*A on rank 0
    if (rank == 0) {
        //omp_set_dynamic(0); // Explicitly disable dynamic teams
        //omp_set_num_threads(1);
        //omp_set_nested(1);
        mkl_set_dynamic(0);
        mkl_set_num_threads(my_nthreads);
        // create sym-pos-def matrix B = A' * A
        cblas_dgemm (CblasColMajor,CblasTrans,CblasNoTrans,
                     N,N,N,1,A,N,A,N,0,B,N);
        save = mkl_get_max_threads();
        mkl_set_num_threads(1);
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
    
    //omp_set_num_threads(1);
    t2 = MPI_Wtime();
    
    
    double sum = 0.0;
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            sum += B[i*N+i]; // find trace of B
        }
        printf("Finding B = A'*A using threads (%d) took: %.3f ms, trace(B) = %.6e\n", save, (t2-t1)*1e3, sum);
    }
    
    if (rank == 0) {
        int nb = N > 10 ? 10 : N;
        printf("First %d x %d block of B:\n",nb,nb);
        for (i = 0; i < nb; i++) {
            for (j = 0; j < nb; j++) {
                printf("%9.3f ", B[j*N+i]);
            }
            printf("\n");
        }
    }
    
    if (rank == 0) {
        free(A);
        free(B);
    }
    
    MPI_Finalize(); // finalize MPI
    return 0;
}










