// Compile: gcc -std=gnu99 -march=corei7-avx -fopenmp marco_template.c -o marco_template.exe
// Reference: 1. https://gcc.gnu.org/onlinedocs/cpp/Macros.html
//            2. https://stackoverflow.com/questions/1253934/c-pre-processor-defining-for-generated-function-names

#include <stdio.h>
#include <string.h>

#define MAKE_VEC_FUN_NAME(STR)          \
void vec_func_##STR(                    \
    const double *x, const double *y,   \
    double *__restrict__ z              \
)

#define VEC_FUN_NAME_WITH_PARAM(PARAM)  MAKE_VEC_FUN_NAME(PARAM)

#define VEC_FUN_TEMPLATE(PARAM1, PARAM2)        \
VEC_FUN_NAME_WITH_PARAM(PARAM1 ## _ ## PARAM2)  \
{                                               \
    for (int i = 0; i < PARAM1; i++)            \
    {                                           \
        _Pragma("omp simd")                     \
        for (int j = 0; j < PARAM2; j++)        \
            z[j] += x[i] * y[j];                \
    }                                           \
}

VEC_FUN_TEMPLATE(2, 4)  // This gives you vec_func_2_4(x, y, z)
VEC_FUN_TEMPLATE(2, 8)  // This gives you vec_func_2_8(x, y, z)
VEC_FUN_TEMPLATE(4, 8)  // This gives you vec_func_4_8(x, y, z)

int main()
{
    double x[8] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0};
    double y[8] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
    double z[8];
    
    memset(z, 0, sizeof(double) * 8);
    vec_func_2_4(x, y, z);
    for (int i = 0; i < 8; i++) printf("%.2lf ", z[i]);
    printf("\n");
    
    memset(z, 0, sizeof(double) * 8);
    vec_func_2_8(x, y, z);
    for (int i = 0; i < 8; i++) printf("%.2lf ", z[i]);
    printf("\n");

    memset(z, 0, sizeof(double) * 8);
    vec_func_4_8(x, y, z);
    for (int i = 0; i < 8; i++) printf("%.2lf ", z[i]);
    printf("\n");

    return 0;
}