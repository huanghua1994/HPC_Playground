function [x, fft_t, d_t] = Poisson_FFTsolver(nx, ny, nz, w2_x, w2_y, w2_z, radius, f)
% Credit: Qimen Xu <qimenxu@gatech.edu>
% Solve Poisson equation -\nabla^2 * x = f with period boundary condition 
% \nabla^2 is discreted with order-(2*radius) finite difference
% The FD domain is a cube with equal mesh size on each direction
% Input parameters:
%   n{x, y, z}   : Number of grid points on x, y, z direction
%   x2_{x, y, z} : FD stencil coefficients, length radius+1
%   radius       : FD radius
%   f            : Right-hand side of the equation, size [nx, ny, nz]
% Output parameters:
%   x            : Solution, size [nx, ny, nz]
%   fft_t, d_t   : Timing results for fft and calc d
    timers = zeros(2, 1);

    tic;
    f_hat = fftn(f);
    fft_t1 = toc;

    tic;
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