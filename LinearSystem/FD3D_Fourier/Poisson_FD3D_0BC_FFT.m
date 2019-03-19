% Use of FFT for FD approximation to u for
% u_xx + u_yy + u_zz = 8,  0 < x,y,z < 1
% u(0,y,z) = u(1,y,z) = u(x,0,z) = u(x,1,z) = u(x,y,0) = u(x,y,1) = 0
% Use order-2 finite difference: u_xx = [u(i-1, j, k) + - 2 * u(i, j, k) + u(i+1, j, k)] / h^2

L  = 1;        % Domain size
Ne = 50;       % Extended-domain gridpoints, should be a multipler of 2
h  = 2*L/Ne;   % Grid spacing
m  = Ne/2 - 1; % Number of interior gridpoints in original domain

x = h*(1:m);
y = h*(1:m);
z = h*(1:m);

% Use zero boundary condition, u0y = u1y = ux0 = ux1 = 0
% Assume that we don't have the f value on the boundary, set it as 0
f = 8 * ones(m, m, m);
f(1,:,:) = 0;
f(m,:,:) = 0;
f(:,1,:) = 0;
f(:,m,:) = 0;
f(:,:,1) = 0;
f(:,:,m) = 0;

% Odd extension of f in x direction
g0 = zeros(Ne, m, m);
g0(2:m+1,:,:)  = f;
g0(m+3:Ne,:,:) = -f(m:-1:1,:,:);
% Odd extension of f in y direction
g1 = zeros(Ne, Ne, m);
g1(:,2:m+1,:)  = g0;
g1(:,m+3:Ne,:) = -g0(:,m:-1:1,:);
% Odd extension of f in z direction
g2 = zeros(Ne, Ne, Ne);
g2(:,:,2:m+1)  = g1;
g2(:,:,m+3:Ne) = -g1(:,:,m:-1:1);

ghat = fftn(g2);

[I, J, K] = meshgrid(0:(Ne-1), 0:(Ne-1), 0:(Ne-1));
% Note: in 2D case, mu = (-4/h^2) * (...)
mu = (-4*(m-1)^2/h^2) * (sin(I*pi/Ne).^2 + sin(J*pi/Ne).^2 + sin(K*pi/Ne).^2);
mu(1,1,1) = 1;   % Avoid 0/0 division; vhat(1,1,1) is known a priori to be 0
v = real(ifftn(-ghat ./ mu));
u = v(1:(m+2), 1:(m+2), 1:(m+2)); % Extract u(i,j) from its odd extension

[X, Y, Z] = meshgrid(x, y, z);
Lap = DiscreteLaplacian3D(1, 1, 1, m, m, m, 1, 1);
rhs_f = -f(:) * (h^2);
u_std = Lap \ rhs_f;
u_fft = u(2:(m+1), 2:(m+1), 2:(m+1));
u_fft = u_fft(:);
diff = u_fft - u_std;
err = norm(diff, inf)