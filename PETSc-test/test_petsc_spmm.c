#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "mmio_utils.h"
#include "petscmat.h"

int main(int argc, char **argv)
{
    int my_rank, comm_size, nvec = 8, ntest = 10;

    PetscInitialize(&argc, &argv, NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (argc < 3)
    {
        if (my_rank == 0)
        {
            printf("Usage: mpirun -np %d %s <mtx-filename> <nvec> <(optional) ntest>\n", comm_size, argv[0]);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        CHKERRQ(PetscFinalize());
        return 1;
    }
    nvec = atoi(argv[2]);
    if (nvec < 1) nvec = 8;
    if (argc > 3) ntest = atoi(argv[3]);

    // Read and distribute matrix
    char *mtx_fname = argv[1];
    int nrow, ncol, nnz;
    int *row = NULL, *col = NULL;
    double *val = NULL, st, et;
    Mat spA, B, C;
    PetscRandom rctx;
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
    CHKERRQ(MatCreateFromOptions(PETSC_COMM_WORLD, NULL, bs, PETSC_DECIDE, PETSC_DECIDE, nrow, ncol, &spA));
    st = MPI_Wtime();
    if (my_rank == 0)
    {
        for (int i = 0; i < nnz; i++)
        {
            if (row[i] < 0 || row[i] >= nrow) printf("%d-th nnz invalid row: %d\n", i, row[i]);
            if (col[i] < 0 || col[i] >= ncol) printf("%d-th nnz invalid col: %d\n", i, col[i]);
            CHKERRQ(MatSetValues(spA, 1, &row[i], 1, &col[i], &val[i], INSERT_VALUES));
        }
        free(row);
        free(col);
        free(val);
    }
    CHKERRQ(MatAssemblyBegin(spA, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(spA, MAT_FINAL_ASSEMBLY));
    CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    CHKERRQ(PetscRandomSetType(rctx, PETSCRAND48));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ncol, nvec, NULL, &B));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nrow, nvec, NULL, &C));
    CHKERRQ(MatSetRandom(B, rctx));
    CHKERRQ(MatZeroEntries(C));
    CHKERRQ(PetscRandomDestroy(&rctx));
    et = MPI_Wtime();
    if (my_rank == 0)
    {
        printf("Assemble sparse A and random dense B in %.3f seconds\n", et - st);
        fflush(stdout);
    }

    // Print matrix distribution -- is this correct?
    /*
    for (int i = 0; i < comm_size; i++)
    {
        if (my_rank == i)
        {
            int srow, erow, scol, ecol;
            printf("Rank %3d owns ", my_rank);
            CHKERRQ(MatGetOwnershipRange(spA, &srow, &erow));
            CHKERRQ(MatGetOwnershipRangeColumn(spA, &scol, &ecol));
            printf("spA(%d : %d, :), ", srow, erow - 1);
            CHKERRQ(MatGetOwnershipRange(B, &srow, &erow));
            CHKERRQ(MatGetOwnershipRangeColumn(B, &scol, &ecol));
            printf("B(%d : %d, %d : %d), ", srow, erow - 1, scol, ecol - 1);
            CHKERRQ(MatGetOwnershipRange(C, &srow, &erow));
            CHKERRQ(MatGetOwnershipRangeColumn(C, &scol, &ecol));
            printf("C(%d : %d, %d : %d)\n", srow, erow - 1, scol, ecol - 1);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    */

    // SpMV tests
    for (int i = 0; i <= ntest; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        CHKERRQ(MatMatMult(spA, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));
        et = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("%.6f\n", et - st);
            fflush(stdout);
        }
    }

    CHKERRQ(MatDestroy(&spA));
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(PetscFinalize());
    return 0;
}