#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

#define STREAM_TYPE double

#ifdef USE_AARCH64_SVE
#include <arm_sve.h>
#endif

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 80000000
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

#define NFUNCTION 7

static STREAM_TYPE a[STREAM_ARRAY_SIZE];
static STREAM_TYPE b[STREAM_ARRAY_SIZE];
static STREAM_TYPE c[STREAM_ARRAY_SIZE];
static char *label[NFUNCTION] = {
    "STREAM COPY   c[i] = a[i]            ", 
    "STREAM SCALE  b[i] = s * c[i]        ", 
    "STREAM ADD    c[i] = a[i] + b[i]     ", 
    "STREAM TRIAD  a[i] = b[i] + s * c[i] ", 
    "BLAS-1 DAXPY  a[i] = a[i] + s * b[i] ", 
    "BLAS-1 DDOT   sum(a[i] * b[i])       ",
    "BLAS-1 DNRM2  || a[:] ||_2           "
};
static double max_time[NFUNCTION], min_time[NFUNCTION], avg_time[NFUNCTION];
static STREAM_TYPE dot_sum;
static STREAM_TYPE nrm2_sum;
static STREAM_TYPE scalar = 3;
size_t *thread_displs;

void stream_copy()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #ifdef USE_AARCH64_SVE
        size_t idx = start_idx;
        svbool_t pg = svwhilelt_b64(idx, end_idx);
        svfloat64_t vec_a;
        do
        {
            vec_a = svldnt1(pg, a + idx);
            svstnt1(pg, c + idx, vec_a);
            idx += svcntd();
            pg = svwhilelt_b64(idx, end_idx);
        } while (svptest_any(svptrue_b64(), pg));
        #else
        for (size_t i = start_idx; i < end_idx; i++) c[i] = a[i];
        #endif
    }
}

void stream_scale()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #ifdef USE_AARCH64_SVE
        size_t idx = start_idx;
        svbool_t pg = svwhilelt_b64(idx, end_idx);
        svfloat64_t vec_scalar = svdup_f64_z(pg, scalar);
        svfloat64_t vec_c;
        do
        {
            vec_c = svldnt1(pg, c + idx);
            vec_c = svmul_f64_z(pg, vec_c, vec_scalar);
            svstnt1(pg, b + idx, vec_c);
            idx += svcntd();
            pg = svwhilelt_b64(idx, end_idx);
        } while (svptest_any(svptrue_b64(), pg));
        #else
        for (size_t i = start_idx; i < end_idx; i++) b[i] = scalar * c[i];
        #endif
    }
}

void stream_add()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #ifdef USE_AARCH64_SVE
        size_t idx = start_idx;
        svbool_t pg = svwhilelt_b64(idx, end_idx);
        svfloat64_t vec_a, vec_b, vec_c;
        do
        {
            vec_a = svldnt1(pg, a + idx);
            vec_b = svldnt1(pg, b + idx);
            vec_c = svadd_f64_z(pg, vec_a, vec_b);
            svstnt1(pg, c + idx, vec_c);
            idx += svcntd();
            pg = svwhilelt_b64(idx, end_idx);
        } while (svptest_any(svptrue_b64(), pg));
        #else
        for (size_t i = start_idx; i < end_idx; i++) c[i] = a[i] + b[i];
        #endif
    }
}

void stream_triad()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #ifdef USE_AARCH64_SVE
        size_t idx = start_idx;
        svbool_t pg = svwhilelt_b64(idx, end_idx);
        svfloat64_t vec_scalar = svdup_f64_z(pg, scalar);
        svfloat64_t vec_a, vec_b, vec_c;
        do
        {
            vec_b = svldnt1(pg, b + idx);
            vec_c = svldnt1(pg, c + idx);
            vec_a = svmad_f64_z(pg, vec_c, vec_scalar, vec_b);
            svstnt1(pg, a + idx, vec_a);
            idx += svcntd();
            pg = svwhilelt_b64(idx, end_idx);
        } while (svptest_any(svptrue_b64(), pg));
        #else
        for (size_t i = start_idx; i < end_idx; i++) a[i] = b[i] + scalar * c[i];
        #endif
    }
}

void blas1_daxpy()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #ifdef USE_AARCH64_SVE
        size_t idx = start_idx;
        svbool_t pg = svwhilelt_b64(idx, end_idx);
        svfloat64_t vec_scalar = svdup_f64_z(pg, scalar);
        svfloat64_t vec_a, vec_b;
        do
        {
            vec_a = svldnt1(pg, a + idx);
            vec_b = svldnt1(pg, b + idx);
            vec_a = svmad_f64_z(pg, vec_b, vec_scalar, vec_a);
            svstnt1(pg, a + idx, vec_a);
            idx += svcntd();
            pg = svwhilelt_b64(idx, end_idx);
        } while (svptest_any(svptrue_b64(), pg));
        #else
        for (size_t i = start_idx; i < end_idx; i++) a[i] += scalar * b[i];
        #endif
    }
}

void blas1_ddot()
{
    #pragma omp parallel reduction(+: dot_sum)
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #ifdef USE_AARCH64_SVE
        size_t idx = start_idx;
        svbool_t pg = svwhilelt_b64(idx, end_idx);
        svfloat64_t vec_a, vec_b, vec_s;
        vec_s = svdup_f64_z(pg, 0.0);
        do
        {
            vec_a = svldnt1(pg, a + idx);
            vec_b = svldnt1(pg, b + idx);
            vec_s = svmla_f64_m(pg, vec_s, vec_a, vec_b);
            idx += svcntd();
            pg = svwhilelt_b64(idx, end_idx);
        } while (svptest_any(svptrue_b64(), pg));
        pg = svwhilelt_b64(start_idx, end_idx);
        STREAM_TYPE local_dot_sum = svaddv_f64(svptrue_b64(), vec_s);
        #else
        STREAM_TYPE local_dot_sum = 0;
        for (size_t i = start_idx; i < end_idx; i++) local_dot_sum += a[i] * b[i];
        #endif
        dot_sum += local_dot_sum;
    }
}

void blas1_dnrm2()
{
    #pragma omp parallel reduction(+: nrm2_sum)
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #ifdef USE_AARCH64_SVE
        // Need a 2-way unroll to issue at least 2 load/store instructions 
        // per iteration to fully utilize the memory bandwidth
        size_t vec_len = svcntd();
        size_t mid_idx = (start_idx + end_idx) / 2;
        size_t idx0 = start_idx;
        size_t idx1 = mid_idx;
        svbool_t pg0 = svwhilelt_b64(idx0, mid_idx);
        svbool_t pg1 = svwhilelt_b64(idx1, end_idx);
        svfloat64_t vec_c0, vec_c1, vec_s0, vec_s1;
        vec_s0 = svdup_f64_z(pg0, 0.0);
        vec_s1 = svdup_f64_z(pg1, 0.0);
        do
        {
            vec_c0 = svldnt1(pg0, c + idx0);
            vec_c1 = svldnt1(pg1, c + idx1);
            vec_s0 = svmla_f64_m(pg0, vec_s0, vec_c0, vec_c0);
            vec_s1 = svmla_f64_m(pg1, vec_s1, vec_c1, vec_c1);
            idx0 += vec_len;
            idx1 += vec_len;
            pg0 = svwhilelt_b64(idx0, mid_idx);
            pg1 = svwhilelt_b64(idx1, end_idx);
        } while (svptest_any(svptrue_b64(), svorr_z(svptrue_b64(), pg0, pg1)));
        vec_s0 = svadd_f64_z(svptrue_b64(), vec_s0, vec_s1);
        STREAM_TYPE local_nrm2_sum = svaddv_f64(svptrue_b64(), vec_s0);
        #else
        STREAM_TYPE local_nrm2_sum = 0;
        for (size_t i = start_idx; i < end_idx; i++) local_nrm2_sum += c[i] * c[i];
        #endif
        nrm2_sum += local_nrm2_sum;
    }
}

int main(int argc, char **argv)
{
    int nthread = omp_get_max_threads();
    printf("Size of test arrays      : %zu\n", (size_t) STREAM_ARRAY_SIZE);
    printf("Bytes per array element  : %zu\n", sizeof(STREAM_TYPE));
    printf("Number of threads to use : %d\n", nthread);
    printf("Number of tests to run   : %d\n", NTIMES);
    double MB = (double) STREAM_ARRAY_SIZE * (double) sizeof(STREAM_TYPE) / 1024.0 / 1024.0;
    double GB = MB / 1024.0;
    printf("Memory per array         : %.1f MiB (= %.1f GiB)\n", MB, GB);

    thread_displs = (size_t *) malloc(sizeof(size_t) * (nthread + 1));
    size_t rem = STREAM_ARRAY_SIZE % nthread;
    size_t bs0 = STREAM_ARRAY_SIZE / nthread;
    size_t bs1 = bs0 + 1;
    for (size_t i = 0; i <= nthread; i++)
        thread_displs[i] = (i < rem) ? (bs1 * i) : (bs0 * i + rem);

    // NUMA first touch initialization
    #pragma omp parallel num_threads(nthread)
    {
        int tid = omp_get_thread_num();
        for (size_t i = thread_displs[tid]; i < thread_displs[tid + 1]; i++)
        {
            a[i] = 1.0;
            b[i] = 2.0;
            c[i] = 0.0;
        }
    }

    // Main timing tests
    double start_t, stop_t, used_sec;
    for (int j = 0; j < NFUNCTION; j++)
    {
        max_time[j] = 0.0;
        min_time[j] = 19241112.0;
        avg_time[j] = 0.0;
    }
    for (int k = 0; k <= NTIMES; k++)
    {
        start_t  = get_wtime_sec();
        stream_copy();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[0] = MAX(max_time[0], used_sec);
            min_time[0] = MIN(min_time[0], used_sec);
            avg_time[0] += used_sec;
        }

        start_t  = get_wtime_sec();
        stream_scale();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[1] = MAX(max_time[1], used_sec);
            min_time[1] = MIN(min_time[1], used_sec);
            avg_time[1] += used_sec;
        }

        start_t  = get_wtime_sec();
        stream_add();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[2] = MAX(max_time[2], used_sec);
            min_time[2] = MIN(min_time[2], used_sec);
            avg_time[2] += used_sec;
        }

        start_t  = get_wtime_sec();
        stream_triad();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[3] = MAX(max_time[3], used_sec);
            min_time[3] = MIN(min_time[3], used_sec);
            avg_time[3] += used_sec;
        }

        start_t  = get_wtime_sec();
        blas1_daxpy();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[4] = MAX(max_time[4], used_sec);
            min_time[4] = MIN(min_time[4], used_sec);
            avg_time[4] += used_sec;
        }

        dot_sum = 0;
        start_t  = get_wtime_sec();
        blas1_ddot();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[5] = MAX(max_time[5], used_sec);
            min_time[5] = MIN(min_time[5], used_sec);
            avg_time[5] += used_sec;
        }

        nrm2_sum = 0.0;
        start_t  = get_wtime_sec();
        blas1_dnrm2();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[6] = MAX(max_time[6], used_sec);
            min_time[6] = MIN(min_time[6], used_sec);
            avg_time[6] += used_sec;
        }
    }

    // Print results
    double bytes[NFUNCTION];
    bytes[0] = 2.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[1] = 2.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[2] = 3.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[3] = 3.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[4] = 3.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[5] = 2.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[6] = 1.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    printf("\nFunction                              Best Rate MB/s  Avg time     Min time     Max time\n");
    for (int j = 0; j < NFUNCTION; j++) 
    {
        avg_time[j] /= (double) NTIMES;
        printf(
            "%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j], 
            1.0E-06 * bytes[j] / min_time[j], avg_time[j], min_time[j], max_time[j]
        );
    }
    printf("\n");

    // Check results
    STREAM_TYPE ai = 1.0, bi = 2.0, ci = 0.0, dot = 0.0, nrm2 = 0.0;
    STREAM_TYPE err_a = 0.0, err_b = 0.0, err_c = 0.0, err_d = 0.0, err_n = 0.0, epsilon;
    for (int k = 0; k <= NTIMES; k++)
    {
        ci   = ai;
        bi   = scalar * ci;
        ci   = ai + bi;
        ai   = bi + scalar * ci;
        ai   = ai + scalar * bi;
        dot  = ai * bi * STREAM_ARRAY_SIZE;
        nrm2 = ci * ci * STREAM_ARRAY_SIZE;
    }
    nrm2 = sqrt(nrm2);
    nrm2_sum = sqrt(nrm2_sum);
    for (size_t i = 0; i < STREAM_ARRAY_SIZE; i++)
    {
        err_a += fabs(a[i] - ai);
        err_b += fabs(b[i] - bi);
        err_c += fabs(c[i] - ci);
    }
    err_a /= (STREAM_TYPE) STREAM_ARRAY_SIZE;
    err_b /= (STREAM_TYPE) STREAM_ARRAY_SIZE;
    err_c /= (STREAM_TYPE) STREAM_ARRAY_SIZE;
    err_d  = dot  - dot_sum;
    err_n  = nrm2 - nrm2_sum;
    if (sizeof(STREAM_TYPE) == 4) epsilon = 1.0e-6;
    if (sizeof(STREAM_TYPE) == 8) epsilon = 1.0e-13;
    int nerr = 0;
    if (fabs(err_a / ai) > epsilon)
    {
        printf("Failed validation on array a[], Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", ai, err_a, fabs(err_a / ai));
        nerr++;
    }
    if (fabs(err_b / bi) > epsilon)
    {
        printf("Failed validation on array b[], Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", bi, err_b, fabs(err_b / bi));
        nerr++;
    }
    if (fabs(err_c / ci) > epsilon)
    {
        printf("Failed validation on array c[], Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", ci, err_c, fabs(err_c / ci));
        nerr++;
    }
    if (fabs(err_d / dot) > epsilon)
    {
        printf("Failed validation on dot result, Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", dot, err_d, fabs(err_d / dot));
        nerr++;
    }
    if (fabs(err_n / nrm2) > epsilon)
    {
        printf("Failed validation on dot result, Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", nrm2, err_n, fabs(err_n / nrm2));
        nerr++;
    }
    if (nerr == 0) printf("Solution Validates: avg error less than %e on all three arrays\n", epsilon);

    free(thread_displs);
    return nerr;
}