#ifndef __FD3D_CG_CUDA_H__
#define __FD3D_CG_CUDA_H__

void FD3D_CG_cuda(
    const int nx, const int ny, const int nz, 
    const double res_tol, const int max_iter, const double *b,
    const double *stencil_coefs, double *x
);

#endif
