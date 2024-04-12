#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "mmio_utils.h"
#include "petscmat.h"

int main(int argc, char **argv)
{
    int my_rank, comm_size, ntest = 10;

    PetscInitialize(&argc, &argv, NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (argc < 2)
    {
        if (my_rank == 0)
        {
            printf("Usage: mpirun -np %d %s <mtx-filename> <(optional) ntest>\n", comm_size, argv[0]);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        CHKERRQ(PetscFinalize());
        return 1;
    }
    if (argc > 2) ntest = atoi(argv[2]);

    // Read and distribute matrix
    char *mtx_fname = argv[1];
    int nrow, ncol, nnz;
    int *row = NULL, *col = NULL;
    double *val = NULL, st, et;
    Mat A;
    Vec x, y;
    if (my_rank == 0)
    {
        st = MPI_Wtime();
        mm_read_sparse_RPI(mtx_fname, 0, &nrow, &ncol, &nnz, &row, &col, &val);
        et = MPI_Wtime();
        printf("Read matrix from %s in %.3f seconds\n", mtx_fname, et - st);
        printf("nrow = %d, ncol = %d, nnz = %d\n", nrow, ncol, nnz);
        fflush(stdout);
    }
    MPI_Bcast(&nrow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncol, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz,  1, MPI_INT, 0, MPI_COMM_WORLD);
    int bs = 1;  // "the blocksize (commonly 1)"
    CHKERRQ(MatCreateFromOptions(PETSC_COMM_WORLD, NULL, bs, PETSC_DECIDE, PETSC_DECIDE, nrow, ncol, &A));
    CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, ncol, &x));
    CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, nrow, &y));
    st = MPI_Wtime();
    if (my_rank == 0)
    {
        for (int i = 0; i < nnz; i++)
            CHKERRQ(MatSetValues(A, 1, &row[i], 1, &col[i], &val[i], INSERT_VALUES));
        free(row);
        free(col);
        free(val);
        int *ix = (int *) malloc(sizeof(int) * ncol);
        double *xarr = (double *) malloc(sizeof(double) * ncol);
        for (int i = 0; i < ncol; i++)
        {
            ix[i] = i;
            xarr[i] = (rand() / (double) RAND_MAX) - 0.5;
        }
        CHKERRQ(VecSetValues(x, ncol, ix, xarr, INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));
    et = MPI_Wtime();
    if (my_rank == 0)
    {
        printf("Assemble sparse A and dense x in %.3f seconds\n", et - st);
        fflush(stdout);
    }

    // Print matrix and vector distributions
    int A_srow, A_erow, x_srow, x_erow, y_srow, y_erow;
    CHKERRQ(MatGetOwnershipRange(A, &A_srow, &A_erow));
    CHKERRQ(VecGetOwnershipRange(x, &x_srow, &x_erow));
    CHKERRQ(VecGetOwnershipRange(y, &y_srow, &y_erow));
    for (int i = 0; i < comm_size; i++)
    {
        if (my_rank == i)
        {
            printf(
                "Rank %3d owns A(%d : %d, :), x(%d : %d), y(%d : %d)\n",
                my_rank, A_srow, A_erow - 1, x_srow, x_erow - 1, y_srow, y_erow - 1
            );
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // SpMV tests
    for (int i = 0; i <= ntest; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        CHKERRQ(MatMult(A, x, y));
        et = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("%.3f\n", et - st);
            fflush(stdout);
        }
    }

    CHKERRQ(MatDestroy(&A));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(PetscFinalize());
    return 0;
}