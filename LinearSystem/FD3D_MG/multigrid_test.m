function multigrid_test(grid_sizes, BCs)
% Test geometry multigrid solver for Poisson equation
% Input parameters:
%   grid_sizes : Number of the finite difference grid points, [Nx, Ny, Nz]
%   BCs        : Boundary condition on each direction, [BCx, BCy, BCz]
%                BC* = 0 is periodic, 1 is Dirichlet
% To improve convergence, choose Nx (Ny, Nz) as follows:
% - along periodic dimensions:  Nx divisible by 2 many times
% - along Dirichlet dimensions: Nx particularly good: 15, 31, ...
% - along Dirichlet dimensions: Nx particularly bad:  24, 32, 40

    cell_dims(1) = (grid_sizes(1) - BCs(1)) * 0.4;
    cell_dims(2) = (grid_sizes(2) - BCs(2)) * 0.4;
    cell_dims(3) = (grid_sizes(3) - BCs(3)) * 0.4;
    FDn = 6;
    latVecs = eye(3);   % Not yet handling nonorthogonal lattices

    tic;
    mg = multigrid_setup(cell_dims, grid_sizes, latVecs, BCs, FDn);
    toc;

    rng(0);
    Nd = prod(grid_sizes);
    b = rand(Nd, 1) - 0.5;

    % If all periodic boundaries, then project b such that the system is consistent
    if norm(BCs) == 0
        t = sum(b) / Nd;
        b = b - t;
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
    
    lastA = full(mg.R{nlevel} * mg.A{nlevel} * mg.P{nlevel});
    if (norm(BCs) == 0)
        mg.lastA_pinv = pinv(lastA);
        mg.use_pinv = 1;
    else
        [mg.lastA_L, mg.lastA_U] = lu(lastA);
        mg.use_pinv = 0;
    end
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
                t = mg.A{level} * mg.e{level}; 
            else
                t0 = mg.P{level} * mg.e{level};
                t1 = mg.A{level} * t0;
                t  = mg.R{level} * t1;
            end
            mg.r{level+1} = mg.R{level+1} * (mg.r{level} - t);
        end

        % Solve on the coarsest level
        if (mg.use_pinv == 0)
            mg.e{level+1} = mg.lastA_U \ (mg.lastA_L \ mg.r{mg.nlevel});
        else
            mg.e{level+1} = mg.lastA_pinv * mg.r{mg.nlevel};
        end

        % Upward pass
        for level = mg.nlevel-1:-1:1
            % Prolong the correction
            mg.e{level} = mg.e{level} + mg.P{level+1} * mg.e{level+1}; 
            % Post-smoothing
            if (level == 1)
                t = mg.A{level} * mg.e{level}; 
            else
                t0 = mg.P{level} * mg.e{level};
                t1 = mg.A{level} * t0;
                t  = mg.R{level} * t1;
            end
            mg.e{level} = mg.e{level} + mg.M{level} .* (mg.r{level} - t);
        end
      
        x = x + mg.e{1};
        mg.r{1} = b - mg.A{1} * x;
        fprintf('%2d   %e\n', k, norm(mg.r{1}) / norm(b));
    end
end
