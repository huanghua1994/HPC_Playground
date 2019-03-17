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
	
	MPI_Barrier(MPI_COMM_WORLD);
	
    double t1, t2;
    double *A, *B, *lambda, *M_temp, *A_cp, *B_cp;
    int N;
    
    // read env variable PRINT_MAT to decide whether to print matrix or not
    char *matrix_size = getenv("MAT_SIZE"); 
    if (matrix_size != NULL) 
        N = atoi(matrix_size);
    else 
        N = 0;
    if (N <= 0) N = 2400;
    
    // define matrix size
    //N = 5000;
    if (rank == 0) printf("Matrix size = %d x %d\n",N,N);

    // define the matrix
    if (rank == 0) {
        srand(1);
        M_temp = (double *)malloc(N*N*sizeof(double));
        A = (double *)malloc(N*N*sizeof(double));
        B = (double *)malloc(N*N*sizeof(double));
        lambda = (double *)malloc(N*sizeof(double));
        
        for (j = 0; j < N; j++) {
            for (i = 0; i < N; i++) {
                A[j*N+i] = rand() / (double)RAND_MAX - 0.5;
                M_temp[j*N+i] = rand() / (double)RAND_MAX -0.5;
            }
        }
        // make A symmetric
        for (j = 0; j < N; j++) {
            for (i = 0; i < j; i++) {
                A[j*N+i] = A[i*N+j];    
            }
        }
        // create sym-pos-def matrix B = M_temp' * M_temp
        cblas_dgemm (CblasColMajor,CblasTrans,CblasNoTrans,
                     N,N,N,1,M_temp,N,M_temp,N,0,B,N);
    }
    
    // keep a copy of A and B
    if (rank == 0) {
        A_cp = (double *)malloc(N*N*sizeof(double));
        B_cp = (double *)malloc(N*N*sizeof(double));
        for (i = 0; i < N*N; i++) {
            A_cp[i] = A[i];
            B_cp[i] = B[i];
        }
    }
    
    int print_flag = -1;
    
    // read env variable PRINT_MAT to decide whether to print matrix or not
    char *print_matrix = getenv("PRINT_MAT"); 
    if (print_matrix != NULL) 
        print_flag = atoi(print_matrix);
    else 
        print_flag = 0;
    if (print_flag != 1) print_flag = 0;
    
    // print matrix out if user specifies
    if (print_flag == 1 & rank == 0) {
        // print out matrix A and matrix B
        printf("A = \n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%8.5f ",A[j*N+i]);
            }
            printf("\n");
        }
        printf("B = \n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%8.5f ",B[j*N+i]);
            }
            printf("\n");
        }
    }
    
    // check env for number of threads to use
    int my_nthreads = 1;
    char *env_ntheads = getenv("NTHREADS"); 
    if (env_ntheads != NULL) 
        my_nthreads = atoi(env_ntheads);
    if (my_nthreads < 1) my_nthreads = 1;
    
    
    // use LAPACK routine to solve generalized eigenproblem Ax = lambda*B*x
    /*MKL_DYNAMIC=false*/
    /*OMP_NESTED=true*/
    /*OMP_NUM_THREADS=2 MKL_NUM_THREADS=4*/
    //omp_set_max_active_levels(2) ;
    
    
    /**  Implementation one **/
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    if (rank == 0) {
        omp_set_dynamic(0); // Explicitly disable dynamic teams
        omp_set_num_threads(1);
        omp_set_nested(1);
        mkl_set_dynamic(0);
        mkl_set_num_threads(4);
        //#pragma omp parallel num_threads(1)
        #pragma omp parallel
        {
            LAPACKE_dsygv(LAPACK_COL_MAJOR,1,'V','U',N,A,N,B,N,lambda);                 
        }
        mkl_set_num_threads(1);
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    t2 = MPI_Wtime();
    */
    
    
    /**  Implementation two **/
    MPI_Request req;
    //MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    if (rank == 0) {
        //omp_set_dynamic(0); // Explicitly disable dynamic teams
        //omp_set_num_threads(1);
        //omp_set_nested(1);
        int save = mkl_get_max_threads();
        mkl_set_dynamic(0);
        mkl_set_num_threads(my_nthreads);
        printf("Default mkl threads = %d, now set to %d\n",save,my_nthreads);
        LAPACKE_dsygv(LAPACK_COL_MAJOR,1,'V','U',N,A,N,B,N,lambda);                 
        mkl_set_dynamic(1);
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
    
    
    
    
    t2 = MPI_Wtime();
    

    
    if (print_flag == 1 & rank == 0) {
        // print out eigenvalues
        printf("lambda = \n");
        for (i = 0; i < N; i++) 
            printf("%8.5f \n",lambda[i]);
            
        // print out eigenvectors
        printf("Q = \n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%8.5f ",A[j*N+i]);
            }
            printf("\n");
        }
    }
    
    
    if (!rank) {
        printf("first ten lambda (enable threading) = \n");
        for (i = 0; i < min(10,N); i++) 
            printf("%8.5f \n",lambda[i]);
    }
    
    
    //time_t current_time = time(NULL);
    //char *c_time_str = ctime(&current_time);
    if (rank == 0) {
        printf("------------------------------------------------------\n");
        //printf("Date: %.24s\n", c_time_str);
        printf("Time for solving generalized eigenproblem: %8.3f ms\n",(t2-t1)*1e3);
        printf("------------------------------------------------------\n");
    }
    
    
    // solve the problem again on single process without threading (as reference answer)
    if (rank == 0) {
        t1 = MPI_Wtime();
        //omp_set_num_threads(1);
        mkl_set_dynamic(0);
        mkl_set_num_threads(1);
        printf("New number of mkl threads = %d\n",mkl_get_max_threads());
        LAPACKE_dsygv(LAPACK_COL_MAJOR,1,'V','U',N,A_cp,N,B_cp,N,lambda); 
        t2 = MPI_Wtime();
        printf("Time for solving the same problem on single thread: %.3f ms\n",(t2-t1)*1e3);
        printf("first ten lambda_ref (without threading) = \n");
        for (i = 0; i < min(10,N); i++) 
            printf("%8.5f \n",lambda[i]);
    }
    
    
    
    
    if (rank == 0) {
        free(A);
        free(B);
        free(lambda);
        free(M_temp);
        free(A_cp);
        free(B_cp);
    }
    
    MPI_Finalize(); // finalize MPI
}










