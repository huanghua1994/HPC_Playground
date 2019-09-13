#ifndef __X86_INTRIN_WRAPPER_H__
#define __X86_INTRIN_WRAPPER_H__

// Reference: https://software.intel.com/sites/landingpage/IntrinsicsGuide/

#include <x86intrin.h>

/*
    Naming: intrin_<operation name>_<type surfix>, type surfix _s is for float,  _d is for double 
    intrin_zero_*    : Set an intrinsic vector to zero
    intrin_set1_*    : Set all lanes of an intrinsic vector to a value
    intrin_bcast_*   : Set all lanes of an intrinsic vector to the 1st value in an address
    intrin_load_*    : Load an intrinsic vector from an address aligned to required bits
    intrin_loadu_*   : Load an intrinsic vector from an address which may not aligned to required bits
    intrin_store_*   : Store an intrinsic vector to an address aligned to required bits
    intrin_storeu_*  : Store an intrinsic vector to an address which may not aligned to required bits
    intrin_add_*     : Add two intrinsic vectors a + b
    intrin_sub_*     : Subtract intrinsic vector a by b
    intrin_mul_*     : Multiply two intrinsic vectors a * b
    intrin_div_*     : Divide intrinsic vector a by b
    intrin_sqrt_*    : Return the square root of an intrisic vector's each lane
    intrin_fmadd_*   : Fused Multiply-Add intrinsic vectors a * b + c
    intrin_fmsub_*   : Fused Multiply-Sub intrinsic vectors a * b - c
    intrin_max_*     : Return each lane's maximum values of two intrinsic vectors 
    intrin_min_*     : Return each lane's minimum values of two intrinsic vectors 
    intrin_cmp_eq_*  : Return in each lane if a == b
    intrin_cmp_neq_* : Return in each lane if a != b
    intrin_cmp_lt_*  : Return in each lane if a <  b
    intrin_cmp_le_*  : Return in each lane if a <= b
    intrin_cmp_gt_*  : Return in each lane if a >  b
    intrin_cmp_ge_*  : Return in each lane if a >= b
*/ 


#ifdef __AVX__

#define SIMD_LEN_S 8
#define SIMD_LEN_D 4
#define vec_t_s __m256
#define vec_t_d __m256d

inline __m256  intrin_zero_s() { return _mm256_setzero_ps(); }
inline __m256d intrin_zero_d() { return _mm256_setzero_pd(); }

inline __m256  intrin_set1_s(const float  a)  { return _mm256_set1_ps(a); }
inline __m256d intrin_set1_d(const double a)  { return _mm256_set1_pd(a); }

inline __m256  intrin_bcast_s(float  const *a)  { return _mm256_broadcast_ss(a); }
inline __m256d intrin_bcast_d(double const *a)  { return _mm256_broadcast_sd(a); }

inline __m256  intrin_load_s(float  const *a)   { return _mm256_load_ps(a); }
inline __m256d intrin_load_d(double const *a)   { return _mm256_load_pd(a); }

inline __m256  intrin_loadu_s(float  const *a)  { return _mm256_loadu_ps(a); }
inline __m256d intrin_loadu_d(double const *a)  { return _mm256_loadu_pd(a); }

inline void intrin_store_s(float  *a, const __m256  b)  { _mm256_store_ps(a, b); }
inline void intrin_store_d(double *a, const __m256d b)  { _mm256_store_pd(a, b); }

inline void intrin_storeu_s(float  *a, const __m256  b) { _mm256_storeu_ps(a, b); }
inline void intrin_storeu_d(double *a, const __m256d b) { _mm256_storeu_pd(a, b); }

inline __m256  intrin_add_s(const __m256  a, const __m256  b) { return _mm256_add_ps(a, b); }
inline __m256d intrin_add_d(const __m256d a, const __m256d b) { return _mm256_add_pd(a, b); }

inline __m256  intrin_sub_s(const __m256  a, const __m256  b) { return _mm256_sub_ps(a, b); }
inline __m256d intrin_sub_d(const __m256d a, const __m256d b) { return _mm256_sub_pd(a, b); }

inline __m256  intrin_mul_s(const __m256  a, const __m256  b) { return _mm256_mul_ps(a, b); }
inline __m256d intrin_mul_d(const __m256d a, const __m256d b) { return _mm256_mul_pd(a, b); }

inline __m256  intrin_div_s(const __m256  a, const __m256  b) { return _mm256_div_ps(a, b); }
inline __m256d intrin_div_d(const __m256d a, const __m256d b) { return _mm256_div_pd(a, b); }

inline __m256  intrin_sqrt_s(const __m256  a) { return _mm256_sqrt_ps(a); }
inline __m256d intrin_sqrt_d(const __m256d a) { return _mm256_sqrt_pd(a); }

#ifdef __AVX2__
inline __m256  intrin_fmadd_s(const __m256  a, const __m256  b, const __m256  c) 
{ return _mm256_fmadd_ps(a, b, c); }
inline __m256d intrin_fmadd_d(const __m256d a, const __m256d b, const __m256d c) 
{ return _mm256_fmadd_pd(a, b, c); }

inline __m256  intrin_fmsub_s(const __m256  a, const __m256  b, const __m256  c) 
{ return _mm256_fmsub_ps(a, b, c); }
inline __m256d intrin_fmsub_d(const __m256d a, const __m256d b, const __m256d c) 
{ return _mm256_fmsub_pd(a, b, c); }
#else
inline __m256  intrin_fmadd_s(const __m256  a, const __m256  b, const __m256  c) 
{ return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
inline __m256d intrin_fmadd_d(const __m256d a, const __m256d b, const __m256d c) 
{ return _mm256_add_pd(_mm256_mul_pd(a, b), c); }

inline __m256  intrin_fmsub_s(const __m256  a, const __m256  b, const __m256  c) 
{ return _mm256_sub_ps(_mm256_mul_ps(a, b), c); }
inline __m256d intrin_fmsub_d(const __m256d a, const __m256d b, const __m256d c) 
{ return _mm256_sub_pd(_mm256_mul_pd(a, b), c); }
#endif // End of #ifdef __AVX2__

inline __m256  intrin_max_s(const __m256  a, const __m256  b) { return _mm256_max_ps(a, b); }
inline __m256d intrin_max_d(const __m256d a, const __m256d b) { return _mm256_max_pd(a, b); }

inline __m256  intrin_min_s(const __m256  a, const __m256  b) { return _mm256_min_ps(a, b); }
inline __m256d intrin_min_d(const __m256d a, const __m256d b) { return _mm256_min_pd(a, b); }

inline __m256  intrin_cmp_eq_s(const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OS); }
inline __m256d intrin_cmp_eq_d(const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OS); }

inline __m256  intrin_cmp_neq_s(const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OS); }
inline __m256d intrin_cmp_neq_d(const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_OS); }

inline __m256  intrin_cmp_lt_s(const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
inline __m256d intrin_cmp_lt_d(const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }

inline __m256  intrin_cmp_le_s(const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
inline __m256d intrin_cmp_le_d(const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }

inline __m256  intrin_cmp_gt_s(const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_GT_OS); }
inline __m256d intrin_cmp_gt_d(const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GT_OS); }

inline __m256  intrin_cmp_ge_s(const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_GE_OS); }
inline __m256d intrin_cmp_ge_d(const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GE_OS); }

#endif  // End of #ifdef __AVX__

#endif
