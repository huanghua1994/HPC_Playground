clear all;

% Finite difference domain 
L1  = 23.34; 
L2  = 23.34; 
L3  = 23.34;
N1  = 36; 
N2  = 36; 
N3  = 36;
N   = N1 * N2 * N3;
% Mesh sizes
dx  = L1 / N1;
dy  = L2 / N2;
dz  = L3 / N3;
dx2 = dx * dx;
dy2 = dy * dy;
dz2 = dz * dz;
% Finite difference weights
FDn = 1;
w2 = zeros(1,FDn+1); 
for k=1:FDn
    w2(k+1) = (2*(-1)^(k+1))*(factorial(FDn)^2) / ...
              (k*k*factorial(FDn-k)*factorial(FDn+k));
    w2(1) = w2(1)-2*(1/(k*k));
end
w2_x = w2 / dx2;
w2_y = w2 / dy2;
w2_z = w2 / dz2;


% Right hand side of the poisson equation
f = dlmread('f_rhs.txt');
f = reshape(f, [N1, N2, N3]);

%------------------------------------------------------
% Solve the 3D Poisson equation using FFT
%------------------------------------------------------
[u_fft, fft_t, d_t] = Poisson_FD3D_PBC_FFT_Solver(N1, N2, N3, w2_x, w2_y, w2_z, FDn, f);
u_fft = u_fft(:);  % flatten the result for comparison
fprintf('Poisson_FD3D_PBC_FFT_Solver done, fft time = %f, d_hat time = %f, total time = %f\n', fft_t, d_t, fft_t + d_t);


%------------------------------------------------------
% Solve the 3D Poisson equation using FD (GMRES solver)
%------------------------------------------------------
f = f(:); % Reshape f to a vector
t_fd = tic;
% Lap has an eigenvalue 0, it's not full-rank. Cannot use CG to solve it.
Lap = DiscreteLaplacian3D(L1, L2, L3, N1, N2, N3, FDn, 2);
u_fd = gmres(Lap, -f, 10, 1e-10, min(1000, N));
fprintf('GMRES solver done, used time = %f\n', toc(t_fd));
% Shift the results by a constant
u_fd = u_fd - sum(u_fd) / N;


% Plot the difference
fprintf('||u_{fft} - u_{fd}||_2 = %e\n', norm(u_fd - u_fft));
x = 1 : N;
semilogy(1 : N, abs(u_fft - u_fd)), grid on
xlabel('x'); ylabel('log_{10} |u_{fft} - u_{fd}|');