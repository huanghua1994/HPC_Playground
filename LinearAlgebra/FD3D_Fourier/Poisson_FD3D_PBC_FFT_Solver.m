function [x, fft_t, d_t] = Poisson_FD3D_PBC_FFT_Solver(nx, ny, nz, w2_x, w2_y, w2_z, radius, f)
% Credit: Qimen Xu <qimenxu@gatech.edu>
% Solve Poisson equation -\nabla^2 * x = f with period boundary condition 
% \nabla^2 is discretized with order-(2*radius) finite difference
% The FD domain is a cube with equal mesh size on each direction
% Input parameters:
%   n{x, y, z}   : Number of grid points on x, y, z direction
%   w2_{x, y, z} : FD stencil coefficients, length radius+1
%   radius       : FD radius
%   f            : Right-hand side of the equation, size [nx, ny, nz]
% Output parameters:
%   x            : Solution, size [nx, ny, nz]
%   fft_t, d_t   : Timing results for fft and calc d
    tic;
    f_hat = fftn(f);
    fft_t1 = toc;

    tic;
    ix_s = floor(-nx/2)+2;
    ix_e = floor( nx/2)+1;
    iy_s = floor(-ny/2)+2;
    iy_e = floor( ny/2)+1;
    iz_s = floor(-nz/2)+2;
    iz_e = floor( nz/2)+1;
    cos_ix = zeros(nx * radius, 1);
    cos_iy = zeros(ny * radius, 1);
    cos_iz = zeros(nz * radius, 1);
    for p = 1 : radius
        tmp_x = 2 * pi * p / nx;
        tmp_y = 2 * pi * p / ny;
        tmp_z = 2 * pi * p / nz;
        for ix = ix_s : ix_e
            ix1 = ix - ix_s;
            cos_ix(ix1*radius+p) = cos((ix-1)*tmp_x)*w2_x(p+1);
        end
        for iy = iy_s : iy_e
            iy1 = iy - iy_s;
            cos_iy(iy1*radius+p) = cos((iy-1)*tmp_y)*w2_y(p+1);
        end
        for iz = iz_s : iz_e
            iz1 = iz - iz_s;
            cos_iz(iz1*radius+p) = cos((iz-1)*tmp_z)*w2_z(p+1);
        end
    end
    
    count = 1;
    d_hat = zeros(nx, ny, nz);
    w2_diag = w2_x(1) + w2_y(1) + w2_z(1);
    for iz = [1:iz_e, iz_s:0]
        iz1 = (iz - iz_s) * radius;
        for iy = [1:iy_e, iy_s:0]
            iy1 = (iy - iy_s) * radius;
            for ix = [1:ix_e, ix_s:0]
                ix1 = (ix - ix_s) * radius;
                res = 0;
                for p = 1 : radius
                    res = res + (cos_ix(ix1+p) + cos_iy(iy1+p) + cos_iz(iz1+p)); 
                end
                res = -2 * res - w2_diag;
                d_hat(count) = res;
                count = count + 1;
            end
        end
    end
    
    %{
    count = 1;
    d_hat = zeros(nx, ny, nz);
    w2_diag = w2_x(1) + w2_y(1) + w2_z(1);
    for k3 = [1:floor(nz/2)+1, floor(-nz/2)+2:0]
        for k2 = [1:floor(ny/2)+1, floor(-ny/2)+2:0]
            for k1 = [1:floor(nx/2)+1, floor(-nx/2)+2:0]
                d_hat(count) = -w2_diag;
                for p = 1:radius
                    d_hat(count) = d_hat(count) - 2 * ...
                        (  cos(2*pi*(k1-1)*p/nx)*w2_x(p+1) ...
                         + cos(2*pi*(k2-1)*p/ny)*w2_y(p+1) ...
                         + cos(2*pi*(k3-1)*p/nz)*w2_z(p+1));
                end
                count = count + 1;
            end
        end
    end
    %}
    
    d_hat(1) = 1;
    x_hat = f_hat ./ d_hat;
    x_hat(1) = 0;
    d_t = toc;

    tic;
    x = ifftn(x_hat);
    fft_t2 = toc;
    fft_t = fft_t1 + fft_t2;

    % Shift the results by a constant
    x = real(x) - sum(sum(sum(real(x)))) / (nx * ny * nz);
end