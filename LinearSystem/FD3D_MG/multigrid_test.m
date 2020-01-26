function multigrid_test
    % To improve convergence, choose number of grid points Nx (or Ny, Nz) as follows:
    % - along periodic dimensions: Nx divisible by 2 many times
    % - along Dirichlet dimensions: Nx particularly good: 15, 31, ...
    % - along Dirichlet dimensions: Nx particularly bad:  24, 32, 40
    Nx = 48;
    Ny = 48;
    Nz = 50;

    BCs = [0 0 1];  % 0 - periodic, 1 - Dirichlet

    L1  = (Nx-BCs(1))*0.4;
    L2  = (Ny-BCs(2))*0.4;
    L3  = (Nz-BCs(3))*0.4;
    FDn = 6;

    cell_dims  = [L1 L2 L3];
    grid_sizes = [Nx Ny Nz];
    latVecs    = eye(3);   % Not yet handling nonorthogonal lattices

    tic;
    mg = multigrid_setup(cell_dims, grid_sizes, latVecs, BCs, FDn);
    toc;

    rng(0);
    b = rand(Nx * Ny * Nz, 1) - 0.5;

    % If all periodic boundaries, then project b such that the system is consistent
    if norm(BCs) == 0
        e = ones(size(b));
        e = e / norm(e);
        b = b - e * (e' * b);
    end

    x0 = zeros(size(b));  % initial guess
    tic;
    x  = multigrid_solve(mg, b, x0);
    toc;

    % Solve using perturbed b and previous solution as guess
    % b = b + 0.01 * randn(size(b));
    % x0 = x; 
    % x = multigrid_solve(mg, b, x0);
end

function mg = multigrid_setup(cell_dims, grid_sizes, latVecs, BCs, FDn)
    % Set up the multigrid method for the given FD mesh configuration.
    Nx = grid_sizes(1);
    Ny = grid_sizes(2);
    Nz = grid_sizes(3);
    Nd = Nx * Ny * Nz;
    mg.vec_len(1) = Nd;

    nlevel = 1;
    while (Nx > 7 && Ny > 7 && Nz > 7)
        nlevel = nlevel + 1;
        [mg.A{nlevel}, A_rowptr, A_col, A_val]     = gen_fd_lap_orth(cell_dims, [Nx Ny Nz], BCs, FDn);
        [mg.R{nlevel}, mg.P{nlevel}, mg.M{nlevel}] = gen_R_P_diag_RAP([Nx Ny Nz], BCs, A_rowptr, A_col, A_val);

        Nx = floor(Nx / 2);
        Ny = floor(Ny / 2);
        Nz = floor(Nz / 2);
        Nd = Nx * Ny * Nz;
        mg.vec_len(nlevel) = Nd;
    end
    mg.nlevel = nlevel;

    mg.A{1} = mg.A{2};
    mg.M{1} = ones(size(mg.A{1}, 1), 1) .* (0.75 / mg.A{1}(1, 1));
    
    for level = 1 : nlevel
        mg.e{level} = zeros(mg.vec_len(level), 1);
        mg.r{level} = zeros(mg.vec_len(level), 1);
    end
    
    [mg.lastA_L, mg.lastA_U] = lu(full(mg.R{nlevel} * mg.A{nlevel} * mg.P{nlevel}));
end

function x = multigrid_solve(mg, b, x0)
    % Solve using the multigrid method with right-hand side vector b
    % and initial guess x0. The structure mg is from multigrid_setup.
    x = x0;
    mg.r{1} = b - mg.A{1} * x;

    % Take 40 V-cycles
    for k = 1 : 40
        % Downward pass
        for level = 1:mg.nlevel-1
            % Pre-smoothing
            mg.e{level} = mg.M{level} .* mg.r{level}; 
            % Restrict the residual
            if (level == 1)
                mg.r{level+1} = mg.R{level+1} * (mg.r{level} - mg.A{level} * mg.e{level}); 
            else
                t0 = mg.P{level} * mg.e{level};
                t1 = mg.A{level} * t0;
                t2 = mg.R{level} * t1;
                mg.r{level+1} = mg.R{level+1} * (mg.r{level} - t2);
            end
        end

        % Solve on the coarsest level
        mg.e{level+1} = mg.lastA_U \ (mg.lastA_L \ mg.r{mg.nlevel});

        % Upward pass
        for level = mg.nlevel-1:-1:1
            % Prolong the correction
            mg.e{level} = mg.e{level} + mg.P{level+1} * mg.e{level+1}; 
            % Post-smoothing
            if (level == 1)
                mg.e{level} = mg.e{level} + mg.M{level} .* (mg.r{level} - mg.A{level} * mg.e{level}); 
            else
                t0 = mg.P{level} * mg.e{level};
                t1 = mg.A{level} * t0;
                t2 = mg.R{level} * t1;
                mg.e{level} = mg.e{level} + mg.M{level} .* (mg.r{level} - t2);
            end
        end
      
        x = x + mg.e{1};
        mg.r{1} = b - mg.A{1} * x;
        fprintf('%2d   %e\n', k, norm(mg.r{1}) / norm(b));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
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
%}
