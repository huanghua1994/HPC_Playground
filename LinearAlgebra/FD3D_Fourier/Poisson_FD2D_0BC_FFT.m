% The original MATLAB file is pois_FD_FFT_2D.m in https://atmos.washington.edu/~breth/classes/AM585/

% Use of FFT for FD approximation to u for
% u_xx + u_yy = 4,  0 < x,y < 1
% u(0,y) = u(1,y) = u(x,0) = u(x,1) = 0
% Use order-2 finite difference: u_xx = [u(i-1, j) + - 2* u(i, j) + u(i+1, j)] / h^2

L  = 1;        % Domain size
Ne = 100;      % Extended-domain gridpoints, should be a multipler of 2
h  = 2*L/Ne;   % Grid spacing
m  = Ne/2 - 1; % Number of interior gridpoints in original domain

x = h*(1:m);
y = h*(1:m);

% Use zero boundary condition, u0y = u1y = ux0 = ux1 = 0
% Assume that we don't have the f value on the boundary, set it as 0
f = 4*ones(m,m);
f(:,1) = 0;
f(:,m) = 0;
f(1,:) = 0;
f(m,:) = 0;

% Odd extension of f in x
g0 = zeros(Ne, m);
g0(2:m+1, :)  = f;
g0(m+3:Ne, :) = -f(m:-1:1, :);
% Odd extension of f in y direction
g1 = zeros(Ne, Ne);
g1(:, 2:m+1)  = g0;
g1(:, m+3:Ne) = -g0(:, m:-1:1);

ghat = fft2(g1);
[I, J] = meshgrid(0:(Ne-1),0:(Ne-1));
mu = (4/h^2)*(sin(I*pi/Ne).^2 + sin(J*pi/Ne).^2);
mu(1,1) = 1;   % Avoid 0/0 division; vhat(1,1) is known a priori to be 0
v = real(ifft2(-ghat./mu));
u = v(1:(m+2),1:(m+2)); % Extract u(i,j) from its odd extension

%  Plot out solution in interior and print out max-norm error
[X, Y] = meshgrid(x,y);
Lap   = delsq(numgrid('S', m + 2));
rhs_f = -f(:) * (h^2);
u_std = Lap \ rhs_f;
u_fft = u(2:(m+1), 2:(m+1));
u_fft = u_fft(:);
err   = norm(u_fft - u_std, inf)
surf(X,Y,u(2:(m+1),2:(m+1)))
xlabel('x'), ylabel('y'), zlabel('u')
title('FD/FFT for 2D Poisson Eqn')