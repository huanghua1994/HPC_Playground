function multigrid_test

% To improve convergence, choose number of grid points Nx (or Ny, Nz) as follows:
% - along periodic dimensions: Nx divisible by 2 many times
% - along Dirichlet dimensions: Nx particularly good: 15, 31, ...
% - along Dirichlet dimensions: Nx particularly bad:  24, 32, 40

Nx = 48;
Ny = 48;
Nz = 50;

BCs = [0 0 1]; % 0 means periodic, 1 means Dirichlet

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
Nx = gridsizes(1);
Ny = gridsizes(2);
Nz = gridsizes(3);

level = 1;
while (Nx > 7 && Ny > 7 && Nz > 7)
  mg.P{level+1} = prolong(Nx, Ny, Nz, BCs);
  mg.R{level+1} = 0.125 * mg.P{level+1}';
  [mg.A{level+1}, rowptr, colidx, val] = gen_fd_lap_orth(cellDims, [Nx Ny Nz], BCs, FDn);  
  mg.M{level+1} = get_RAP_diag([Nx Ny Nz], BCs, rowptr, colidx, val);

  Nx = floor(Nx / 2);
  Ny = floor(Ny / 2);
  Nz = floor(Nz / 2);

  level = level + 1;
end

mg.A{1} = mg.A{2};
mg.M{1} = ones(size(mg.A{1}, 1), 1) .* (0.75 / mg.A{1}(1, 1));

mg.numlevels = level;
[mg.lastA_L, mg.lastA_U] = lu(full(mg.R{level} * mg.A{level} * mg.P{level}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = multigrid_solve(mg, b, x0)
% Solve using the multigrid method with right-hand side vector b
% and initial guess x0.  The structure mg is from multigrid_setup.

x = x0;
r{1} = b - mg.A{1}*x;

% take 40 V-cycles
tic;
for k = 1:40
  % Downward pass
  for level = 1:mg.numlevels-1
    % Pre-smoothing
    e{level} = mg.M{level} .* r{level}; 
    % Restrict the residual
    if (level == 1)
      r{level+1} = mg.R{level+1} * (r{level} - mg.A{level} * e{level}); 
    else
      t0 = mg.P{level} * e{level};
      t1 = mg.A{level} * t0;
      t2 = mg.R{level} * t1;
      r{level+1} = mg.R{level+1} * (r{level} - t2);
    end
  end

  % Solve on the coarsest level
  e{level+1} = mg.lastA_U \ (mg.lastA_L \ r{mg.numlevels});

  % Upward pass
  for level = mg.numlevels-1:-1:1
    % Prolong the correction
    e{level} = e{level} + mg.P{level+1} * e{level+1}; 
    % Post-smoothing
    if (level == 1)
      e{level} = e{level} + mg.M{level} .* (r{level} - mg.A{level} * e{level}); 
    else
      t0 = mg.P{level} * e{level};
      t1 = mg.A{level} * t0;
      t2 = mg.R{level} * t1;
      e{level} = e{level} + mg.M{level} .* (r{level} - t2);
    end
  end
  
  x = x + e{1};
  
  r{1} = b - mg.A{1}*x;
  fprintf('%2d   %e\n', k, norm(r{1}) / norm(b));
end
toc;

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

