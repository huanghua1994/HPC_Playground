function multigrid_test1(gridsizes, BCs)

% Along periodic dimensions, must choose N to be even.
% Along Dirichlet dimensions, there is no restriction on N.

cellDims(1) = (gridsizes(1) - BCs(1)) * 0.4;
cellDims(2) = (gridsizes(2) - BCs(2)) * 0.4;
cellDims(3) = (gridsizes(3) - BCs(3)) * 0.4;
%theta = pi/4;
%latVecs = [1 0 0; cos(theta) sin(theta) 0; 0 0 1]; 
latVecs = eye(3); % not yet handling nonorth lattices
FDn = 6;

mg = multigrid_setup(cellDims, gridsizes, latVecs, BCs, FDn);

rng(0);
b = rand(prod(gridsizes),1)-.5;

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

% fprintf('%2d   %e\n', Nx, norm(b-mg.A{1}*x)/norm(b));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mg = multigrid_setup(cellDims, gridsizes, latVecs, BCs, FDn)
% Set up the multigrid method for the given FD mesh configuration.

A = GenDiscreteLaplacian(cellDims, gridsizes, latVecs, BCs, FDn);
mg.A{1} = A;

Nx = gridsizes(1);
Ny = gridsizes(2);
Nz = gridsizes(3);

level = 1;
coarse_size = 7; % select this based on efficiency
while (Nx > coarse_size && Ny > coarse_size && Nz > coarse_size)
  [P Nx Ny Nz] = prolong(Nx, Ny, Nz, BCs);
  R = 0.125*P';

  mg.P{level} = P;
  mg.R{level} = R;
  mg.M{level} = (1/0.75)*diag(diag(A));

  A = R*A*P;
  mg.A{level+1} = A;

  level = level + 1;
end
mg.numlevels = level;

% In the all periodic case, the coarse grid matrix may be rank deficient,
% but the Cholesky factorization still works and can be used to solve the system
mg.chol = chol(-mg.A{mg.numlevels}, 'lower'); % note sign

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = multigrid_solve(mg, b, x0)
% Solve using the multigrid method with right-hand side vector b
% and initial guess x0.  The structure mg is from multigrid_setup.

x = x0;

% take 40 V-cycles
for k = 1:40

  r{1} = b-mg.A{1}*x;

  % downward pass
  for level = 1:mg.numlevels-1
    e{level} = mg.M{level} \ r{level}; % pre-smoothing
    r{level+1} = mg.R{level} * (r{level}-mg.A{level}*e{level}); % restrict the residual
  end

  % solve on the coarsest level
  e{level+1} = - mg.chol' \ (mg.chol \ r{mg.numlevels}); % note sign

  % upward pass
  for level = mg.numlevels-1:-1:1
    e{level} = e{level} + mg.P{level} * e{level+1}; % prolong the correction
    e{level} = e{level} + mg.M{level} \ (r{level}-mg.A{level}*e{level}); % post-smoothing
  end

  x = x + e{1};

  fprintf('%2d   %e\n', k, norm(b-mg.A{1}*x)/norm(b));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P Nx1 Ny1 Nz1] = prolong(Nx, Ny, Nz, BCs)
% 3D prolongator using trilinear interpolation
% Nx = grid dim in x direction at this level (not global Nx), etc.

e = ones(Nx,1); x = spdiags([e 2*e e], -1:1, Nx, Nx); if BCs(1) == 0, x(1,end)=1; x(end,1)=1; end
e = ones(Ny,1); y = spdiags([e 2*e e], -1:1, Ny, Ny); if BCs(2) == 0, y(1,end)=1; y(end,1)=1; end
e = ones(Nz,1); z = spdiags([e 2*e e], -1:1, Nz, Nz); if BCs(3) == 0, z(1,end)=1; z(end,1)=1; end
P = 0.125*kron(kron(z,y),x);

% select coarse grid points
cx = 2:2:Nx;
cy = 2:2:Ny;
cz = 2:2:Nz;

Nx1 = length(cx);
Ny1 = length(cy);
Nz1 = length(cz);

% select columns corresponding to coarse grid points
cpts = zeros(Nx1*Ny1*Nz1, 1);
l = 0;
for i = cz
for j = cy
for k = cx
  l = l + 1;
  cpts(l) = (i-1)*Ny*Nx + (j-1)*Nx + k;
end
end
end
P = P(:,cpts);
