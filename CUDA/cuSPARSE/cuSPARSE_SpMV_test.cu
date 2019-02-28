#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "CUDA_Utils.h"

extern "C" {
#include "cuSPARSE_SpMV_test.h"
}

extern "C"
void cuSPARSE_SpMV_test(
    const int nrows, const int ncols, const int nnz, 
    const int *row_ptr, const int *col, const double *val, 
    const double *x, double *y, const int ntest
)
{
    int nrows1 = nrows + 1;
    const double alpha = 1.0;
    const double beta  = 0.0;
    
    // Allocate memory on device
    int *cu_row_ptr, *cu_col;
    double *cu_val, *cu_x, *cu_y;
    cudaCheck( cudaMalloc(&cu_row_ptr, sizeof(int)    * nrows1) );
    cudaCheck( cudaMalloc(&cu_col,     sizeof(int)    * nnz)    );
    cudaCheck( cudaMalloc(&cu_val,     sizeof(double) * nnz)    );
    cudaCheck( cudaMalloc(&cu_x,       sizeof(double) * ncols)  );
    cudaCheck( cudaMalloc(&cu_y,       sizeof(double) * nrows)  );
    
    // Copy CSR matrix to device
    cudaCheck( cudaMemcpy(cu_row_ptr, row_ptr, sizeof(int)    * nrows1,    cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(cu_col,     col,     sizeof(int)    * nnz,       cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(cu_val,     val,     sizeof(double) * nnz,       cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(cu_x,       x,       sizeof(double) * ncols,     cudaMemcpyHostToDevice) );
    
    // Initialize cuSPARSE
    cusparseHandle_t   cu_SpMV_handle = 0;
    cudaStream_t       cu_SpMV_stream = 0;
    cusparseMatDescr_t cu_SpMV_descr  = 0;
    cudaSparseCheck( cusparseCreate(&cu_SpMV_handle)                                  );
    cudaSparseCheck( cusparseCreateMatDescr(&cu_SpMV_descr)                           );
    cudaSparseCheck( cusparseSetStream(cu_SpMV_handle, cu_SpMV_stream)                );
    cudaSparseCheck( cusparseSetMatType(cu_SpMV_descr, CUSPARSE_MATRIX_TYPE_GENERAL)  );
    cudaSparseCheck( cusparseSetMatIndexBase(cu_SpMV_descr, CUSPARSE_INDEX_BASE_ZERO) );
    
    // Run SpMV on GPU
    cudaEvent_t st, et;
    float SpMV_ms;
    cudaEventCreate(&st);
    cudaEventCreate(&et);
    cudaEventRecord(st, cu_SpMV_stream);
    for (int i = 0; i < ntest; i++)
    {
        cudaSparseCheck(cusparseDcsrmv(
            cu_SpMV_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nrows, ncols, nnz, &alpha, cu_SpMV_descr, cu_val, 
            cu_row_ptr, cu_col, cu_x, &beta, cu_y
        ));
    }
    cudaEventRecord(et, cu_SpMV_stream);
    cudaEventSynchronize(et);
    cudaEventElapsedTime(&SpMV_ms, st, et);
    
    double GFlops, ut;
    GFlops = 2.0 * (double) nnz / 1000000000.0;
    ut = (double) SpMV_ms / (double) (ntest * 1000);
    printf("cuSPARSE  GPU CSR SpMV done, used time = %lf (ms), %lf GFlops\n", ut * 1000.0, GFlops / ut);
    
    // Copy results to host
    cudaCheck( cudaMemcpy(y, cu_y, sizeof(double) * nrows, cudaMemcpyDeviceToHost) );
    
    // Free CUDA resources
    cudaCheck( cudaFree(cu_row_ptr) );
    cudaCheck( cudaFree(cu_col)     );
    cudaCheck( cudaFree(cu_val)     );
    cudaCheck( cudaFree(cu_x)       );
    cudaCheck( cudaFree(cu_y)       );
    cudaSparseCheck( cusparseDestroy(cu_SpMV_handle)        );
    cudaSparseCheck( cusparseDestroyMatDescr(cu_SpMV_descr) );
}

void cuSPARSE_SpMM_test(
    const int nrows, const int ncols, const int nnz, const int X_ncol, 
    const int *row_ptr, const int *col, const double *val, 
    const double *X, const int ldX, double *Y, const int ldY, const int ntest
)
{
    int nrows1 = nrows + 1;
    const double alpha = 1.0;
    const double beta  = 0.0;
    
    // Allocate memory on device
    int *cu_row_ptr, *cu_col;
    double *cu_val, *cu_X, *cu_Y;
    cudaCheck( cudaMalloc(&cu_row_ptr, sizeof(int)    * nrows1) );
    cudaCheck( cudaMalloc(&cu_col,     sizeof(int)    * nnz)    );
    cudaCheck( cudaMalloc(&cu_val,     sizeof(double) * nnz)    );
    cudaCheck( cudaMalloc(&cu_X,       sizeof(double) * ncols * X_ncol)  );
    cudaCheck( cudaMalloc(&cu_Y,       sizeof(double) * nrows * X_ncol)  );
    
    // Copy CSR matrix to device
    cudaCheck( cudaMemcpy(cu_row_ptr, row_ptr, sizeof(int)    * nrows1,         cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(cu_col,     col,     sizeof(int)    * nnz,            cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(cu_val,     val,     sizeof(double) * nnz,            cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(cu_X,       X,       sizeof(double) * ncols * X_ncol, cudaMemcpyHostToDevice) );
    
    // Initialize cuSPARSE
    cusparseHandle_t   cu_SpMM_handle = 0;
    cudaStream_t       cu_SpMM_stream = 0;
    cusparseMatDescr_t cu_SpMM_descr  = 0;
    cudaSparseCheck( cusparseCreate(&cu_SpMM_handle)                                  );
    cudaSparseCheck( cusparseCreateMatDescr(&cu_SpMM_descr)                           );
    cudaSparseCheck( cusparseSetStream(cu_SpMM_handle, cu_SpMM_stream)                );
    cudaSparseCheck( cusparseSetMatType(cu_SpMM_descr, CUSPARSE_MATRIX_TYPE_GENERAL)  );
    cudaSparseCheck( cusparseSetMatIndexBase(cu_SpMM_descr, CUSPARSE_INDEX_BASE_ZERO) );
    
    // Run SpMV on GPU
    cudaEvent_t st, et;
    float SpMM_ms;
    cudaEventCreate(&st);
    cudaEventCreate(&et);
    cudaEventRecord(st, cu_SpMM_stream);
    for (int i = 0; i < ntest; i++)
    {
        cudaSparseCheck(cusparseDcsrmm(
            cu_SpMM_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nrows, X_ncol, ncols, nnz, &alpha, cu_SpMM_descr, 
            cu_val, cu_row_ptr, cu_col, cu_X, ncols, &beta, cu_Y, nrows
        ));
    }
    cudaEventRecord(et, cu_SpMM_stream);
    cudaEventSynchronize(et);
    cudaEventElapsedTime(&SpMM_ms, st, et);
    
    double GFlops, ut;
    GFlops = 2.0 * (double) nnz * (double) X_ncol / 1000000000.0;
    ut = (double) SpMM_ms / (double) (ntest * 1000);
    printf("cuSPARSE  GPU CSR SpMM done, used time = %lf (ms), %lf GFlops\n", ut * 1000.0, GFlops / ut);
    
    // Copy results to host
    cudaCheck( cudaMemcpy(Y, cu_Y, sizeof(double) * nrows * X_ncol, cudaMemcpyDeviceToHost) );
    
    // Free CUDA resources
    cudaCheck( cudaFree(cu_row_ptr) );
    cudaCheck( cudaFree(cu_col)     );
    cudaCheck( cudaFree(cu_val)     );
    cudaCheck( cudaFree(cu_X)       );
    cudaCheck( cudaFree(cu_Y)       );
    cudaSparseCheck( cusparseDestroy(cu_SpMM_handle)        );
    cudaSparseCheck( cusparseDestroyMatDescr(cu_SpMM_descr) );
}
