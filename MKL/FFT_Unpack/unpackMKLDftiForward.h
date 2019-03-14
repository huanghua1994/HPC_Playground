#ifndef __UNPACK_MKL_DFTI_FORWARD_H__
#define __UNPACK_MKL_DFTI_FORWARD_H__

// Unpack MKL 1D FFT CCE output format after 2D FFT forward transform
// Input parameters:
//   nx   : Data length
//   data : MKL FFT 1D forward transform output
// Output parameters:
//   data : Unpacked FFT 1D forward transform result
void unpackMKLDftiForward1DInplace(const int nx, double _Complex *data);

// Unpack MKL 2D FFT CCE output format after 2D FFT forward transform
// Input parameters:
//   n{y, x} : Data dimensions, y is the slowest running index and x is the fastest
//   data    : MKL FFT 2D forward transform output
// Output parameters:
//   data    : Unpacked FFT 2D forward transform result
void unpackMKLDftiForward2DInplace(const int ny, const int nx, double _Complex *data);

// Unpack MKL 3D FFT CCE output format after 3D FFT forward transform
// Input parameters:
//   n{z, y, x} : Data dimensions, z is the slowest running index and x is the fastest
//   data       : MKL FFT 3D forward transform output
// Output parameters:
//   data       : Unpacked FFT 3D forward transform result
void unpackMKLDftiForward3DInplace(const int nz, const int ny, const int nx, double _Complex *data);

#endif
