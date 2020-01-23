function multigrid_test

% To improve convergence, choose number of grid points Nx (or Ny, Nz) as follows:
% - along periodic dimensions: Nx divisible by 2 many times
% - along Dirichlet dimensions: Nx particularly good: 15, 31, ...
% - along Dirichlet dimensions: Nx particularly bad:  24, 32, 40

Nx = 48;
Ny = 50;
Nz = 50;

BCs = [0 1 1]; % 0 means periodic, 1 means Dirichlet

L1 = (Nx-BCs(1))*0.4;
L2 = (Ny-BCs(2))*0.4;
L3 = (Nz-BCs(3))*0.4;

cellDims  = [L1 L2 L3];
gridsizes = [Nx Ny Nz];
latVecs = eye(3); % not yet handling nonorthogonal lattices
FDn = 6;

tic;
mg = multigrid_setup(cellDims, gridsizes, latVecs, BCs, FDn);
toc

rng(0);
b = rand(Nx*Ny*Nz,1)-.5;

% if all periodic boundaries, then project b such that the system is consistent
if norm(BCs) == 0
  e = ones(size(b));
  e = e/norm(e);
  b = b - e*(e'*b);
end

x0 = zeros(size(b)); % initial guess
x = multigrid_solve(mg, b, x0);

% solve using perturbed b and previous solution as guess
% b = b + 0.01*randn(size(b));
% x0 = x; 
% x = multigrid_solve(mg, b, x0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mg = multigrid_setup(cellDims, gridsizes, latVecs, BCs, FDn)
% Set up the multigrid method for the given FD mesh configuration.

%A = GenDiscreteLaplacian(cellDims, gridsizes, latVecs, BCs, FDn);
[A, ~, ~, ~] = gen_fd_lap_orth(cellDims, gridsizes, BCs, FDn);
mg.A{1} = A;
mg.M{1} = 0.75 ./ full(diag(A));

Nx = gridsizes(1);
Ny = gridsizes(2);
Nz = gridsizes(3);

level = 1;
while (Nx > 7 && Ny > 7 && Nz > 7)
  P = prolong(Nx, Ny, Nz, BCs);
  R = 0.125*P';

  mg.P{level} = P;
  mg.R{level} = R;

  % use low order discretization to form coarse matrix
  %Alow = GenDiscreteLaplacian(cellDims, [Nx Ny Nz], latVecs, BCs, 1);
  [Alow, rowptr, colidx, val] = gen_fd_lap_orth(cellDims, [Nx Ny Nz], BCs, 1);
  A = R*Alow*P;
  mg.A{level+1} = A;
  %mg.M{level+1} = 0.75 ./ full(diag(A));
  mg.M{level+1} = get_RAP_diag([Nx Ny Nz], BCs, rowptr, colidx, val);

  Nx = length(2:2:Nx);
  Ny = length(2:2:Ny);
  Nz = length(2:2:Nz);

  level = level + 1;
end
mg.numlevels = level;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = multigrid_solve(mg, b, x0)
% Solve using the multigrid method with right-hand side vector b
% and initial guess x0.  The structure mg is from multigrid_setup.

x = x0;

% take 10 V-cycles
for k = 1:40

  r{1} = b-mg.A{1}*x;

  % downward pass
  for level = 1:mg.numlevels-1
    %e{level} = mg.M{level} \ r{level}; % pre-smoothing
    e{level} = mg.M{level} .* r{level}; % pre-smoothing
    r{level+1} = mg.R{level} * (r{level}-mg.A{level}*e{level}); % restrict the residual
  end

  % solve on the coarsest level
  e{level+1} = mg.A{mg.numlevels} \ r{mg.numlevels};

  % upward pass
  for level = mg.numlevels-1:-1:1
    e{level} = e{level} + mg.P{level} * e{level+1}; % prolong the correction
    %e{level} = e{level} + mg.M{level} \ (r{level}-mg.A{level}*e{level}); % post-smoothing
    e{level} = e{level} + mg.M{level} .* (r{level}-mg.A{level}*e{level}); % post-smoothing
  end

  x = x + e{1};

  fprintf('%2d   %e\n', k, norm(b-mg.A{1}*x)/norm(b));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = prolong(Nx, Ny, Nz, BCs)
% 3D prolongator using trilinear interpolation
% Nx = grid dim in x direction at this level (not global Nx), etc.

e = ones(Nx,1); x = spdiags([e 2*e e], -1:1, Nx, Nx); if BCs(1) == 0, x(1,end)=1; x(end,1)=1; end
e = ones(Ny,1); y = spdiags([e 2*e e], -1:1, Ny, Ny); if BCs(2) == 0, y(1,end)=1; y(end,1)=1; end
e = ones(Nz,1); z = spdiags([e 2*e e], -1:1, Nz, Nz); if BCs(3) == 0, z(1,end)=1; z(end,1)=1; end
p = 0.125*kron(kron(z,y),x);

% select columns corresponding to coarse grid points
len = length(2:2:Nx)*length(2:2:Ny)*length(2:2:Nz);
cpts = zeros(len,1);
l = 0;
for i=2:2:Nz
for j=2:2:Ny
for k=2:2:Nx
  l = l + 1;
  cpts(l) = (i-1)*Ny*Nx + (j-1)*Nx + k;
end
end
end
p = p(:,cpts);

