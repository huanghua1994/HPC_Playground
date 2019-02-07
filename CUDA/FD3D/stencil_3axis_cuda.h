#ifndef __STENCIL_3AXIS_CUDA_H__
#define __STENCIL_3AXIS_CUDA_H__

void stencil_3axis_cuda(
    const double *x0, const int radius, const double *stencil_coefs, 
    const int nx, const int ny, const int nz, double *x1, const int ntest
);

#endif
