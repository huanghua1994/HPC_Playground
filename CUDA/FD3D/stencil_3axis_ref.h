#ifndef __STENCIL_3AXIS_REF_H__
#define __STENCIL_3AXIS_REF_H__

#ifdef __cplusplus
extern "C" {
#endif

void stencil_3axis_ref(
    const double *x0, const int radius, const double *stencil_coefs, 
    const int nx, const int ny, const int nz, double *x1
);

#ifdef __cplusplus
}
#endif

#endif
