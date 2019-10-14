#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define DTYPE double

void Stokes_krnl_eval(
    const DTYPE *coord0, const int ld0, const int n0, 
    const DTYPE *coord1, const int ld1, const int n1, 
    DTYPE *mat, const int ldm 
)
{
    const DTYPE Stokes_eta = 0.5;
    const DTYPE Stokes_a   = 0.25;
    const DTYPE Stokes_C   = 1.0 / (6.0 * M_PI * Stokes_a * Stokes_eta);
    const DTYPE Stokes_Ca3o4 = Stokes_C * Stokes_a * 0.75;
    for (int i = 0; i < n0; i++)
    {
        DTYPE tx = coord0[i];
        DTYPE ty = coord0[i + ld0];
        DTYPE tz = coord0[i + ld0 * 2];
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = tx - coord1[j];
            DTYPE dy = ty - coord1[j + ld1];
            DTYPE dz = tz - coord1[j + ld1 * 2];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            DTYPE r  = sqrt(r2);
            DTYPE inv_r = (r == 0.0) ? 0.0 : 1.0 / r;
            
            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;
            
            DTYPE t1;
            if (r == 0.0) t1 = Stokes_C; 
            else t1 = inv_r * Stokes_Ca3o4;
            
            int base = 3 * i * ldm + 3 * j;
            DTYPE tmp;
            #define krnl(k, l) mat[base + k * ldm + l]
            tmp = t1 * dx;
            krnl(0, 0) = tmp * dx + t1;
            krnl(0, 1) = tmp * dy;
            krnl(0, 2) = tmp * dz;
            tmp = t1 * dy;
            krnl(1, 0) = tmp * dx;
            krnl(1, 1) = tmp * dy + t1;
            krnl(1, 2) = tmp * dz;
            tmp = t1 * dz;
            krnl(2, 0) = tmp * dx;
            krnl(2, 1) = tmp * dy;
            krnl(2, 2) = tmp * dz + t1;
            #undef krnl
        }
    }
}

int main(int argc, char **argv)
{
    const int npt = 4;
    DTYPE coord0[4 * 3] = {
        0, 1, 7, 21, 
        0, 2, 6, 23, 
        0, 3, 9, 25};
    
    DTYPE *mat = malloc(sizeof(DTYPE) * npt * npt * 9);
    Stokes_krnl_eval(
        coord0, npt, npt, 
        coord0, npt, npt,
        mat, npt * 3
    );
    
    for (int i = 0; i < npt * 3; i++)
    {
        DTYPE *mat_i = mat + i * npt * 3;
        for (int j = 0; j < npt * 3; j++) printf("% .3lf ", mat_i[j]);
        printf("\n");
    }
    
    free(mat);
    return 0;
}