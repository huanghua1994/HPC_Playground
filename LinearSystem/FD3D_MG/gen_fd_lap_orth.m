function [Lap, rowptr, colidx, val] = gen_fd_lap_orth(cell_dims, grid_sizes, BCs, FDn)
% Generate finite difference Laplacian matrix for orthogonal
% and equal spacing 3D grid
% Input parameters:
%   cell_dims  : x, y, z direction lengths of the cell
%   grid_sizes : Number of grid points on x, y, z direction
%   BCs        : Boundary condition on x, y, z direction, 0 - periodic, 1 - Dirichlet
%                For Dirichlet, the direction with have grid_sizes[k]-1 FD intervals.
%                For periodic, the  direction with have grid_sizes[k] FD intervals.
%   FDn        : Finite difference order. 7-point stencil is 1, 13-point is 2, etc.
% Output parameter:
%   Lap : A sparse Laplacian matrix. Grid (ix, iy, iz) corresponds to row 
%         ix + (iy-1) * grid_sizes(1) + (iz-1) * grid_sizes(1) * grid_sizes(2)
%   rowptr, colidx, val : Lap stored in the CSR format

    % Finite difference weights for 2nd order derivative
    w2 = zeros(1, FDn+1); 
    c1 = factorial(FDn)^2;
    for k = 1 : FDn
        c2 = 2 * (-1)^(k+1);
        c3 = 1.0 / (k * k);
        c4 = factorial(FDn - k);
        c5 = factorial(FDn + k);
        w2(k+1) = c1 * c2 * c3 / (c4 * c5);
        w2(1) = w2(1) - 2 * c3;
    end
    
    % Cell lengths
    Lx = cell_dims(1);
    Ly = cell_dims(2);
    Lz = cell_dims(3);

    % Boundary conditions in each direction, 0 - periodic, 1 - Dirichlet
    BCx = BCs(1);
    BCy = BCs(2);
    BCz = BCs(3);

    % Number of finite difference grid points
    Nx = grid_sizes(1);
    Ny = grid_sizes(2);
    Nz = grid_sizes(3);
    Nd = Nx * Ny * Nz; 

    % Number of finite difference intervals
    Nintv_x = Nx - BCx;
    Nintv_y = Ny - BCy;
    Nintv_z = Nz - BCz;

    % Mesh spacing and coefficients
    dx = Lx / Nintv_x;
    dy = Ly / Nintv_y;
    dz = Lz / Nintv_z;
    coef_dxx = zeros(2 * FDn + 1, 1);
    coef_dyy = zeros(2 * FDn + 1, 1);
    coef_dzz = zeros(2 * FDn + 1, 1);
    for r = -FDn : FDn
        shift_r = FDn + 1 + r;
        coef_dxx(shift_r) = w2(1 + abs(r)) ./ (dx^2);
        coef_dyy(shift_r) = w2(1 + abs(r)) ./ (dy^2);
        coef_dzz(shift_r) = w2(1 + abs(r)) ./ (dz^2);
    end
    coef_0 = coef_dxx(FDn+1) + coef_dyy(FDn+1) + coef_dzz(FDn+1);

    max_nnz = Nd * (6 * FDn + 1);
    row = zeros(max_nnz, 1);
    col = zeros(max_nnz, 1);
    val = zeros(max_nnz, 1);
    rowptr = zeros(Nd + 1, 1);
    nnz = 0;
    for iz = 1 : Nz
        shift_iz = calc_fd_shift_pos(iz, Nz, BCz, FDn);
        for iy = 1 : Ny
            shift_iy = calc_fd_shift_pos(iy, Ny, BCy, FDn);
            for ix = 1 : Nx
                % (ix, iy, iz)
                curr_row = ix + (iy - 1) * Nx + (iz - 1) * Ny * Nx;
                nnz = nnz + 1;
                row(nnz) = curr_row;
                col(nnz) = curr_row;
                val(nnz) = coef_0;
                rowptr(curr_row) = nnz;
                % (ix +- r, iy, iz)
                % (ix, iy +- r, iz)
                % (ix, iy, iz +- r)
                for r = -FDn : FDn
                    if (r == 0), continue; end
                    shift_r = FDn + 1 + r;
                    iyr  = shift_iy(shift_r);
                    izr  = shift_iz(shift_r);
                    ixr  = -1;
                    ixr0 = ix + r;
                    if ((1 <= ixr0) && (ixr0 <= Nx)), ixr = ixr0;      end
                    if ((ixr0 <  1) && (BCx == 0)),   ixr = ixr0 + Nx; end
                    if ((ixr0 > Nx) && (BCx == 0)),   ixr = ixr0 - Nx; end
                    if (ixr ~= -1)
                        nnz = nnz + 1;
                        row(nnz) = curr_row;
                        col(nnz) = ixr + (iy - 1) * Nx + (iz - 1) * Ny * Nx;
                        val(nnz) = coef_dxx(shift_r);
                    end
                    if (iyr ~= -1)
                        nnz = nnz + 1;
                        row(nnz) = curr_row;
                        col(nnz) = ix + (iyr - 1) * Nx + (iz - 1) * Ny * Nx;
                        val(nnz) = coef_dyy(shift_r);
                    end
                    if (izr ~= -1)
                        nnz = nnz + 1;
                        row(nnz) = curr_row;
                        col(nnz) = ix + (iy - 1) * Nx + (izr - 1) * Ny * Nx;
                        val(nnz) = coef_dzz(shift_r);
                    end
                end
            end
        end
    end
    rowptr(Nd + 1) = nnz + 1;
    
    row = row(1 : nnz);
    col = col(1 : nnz);
    val = val(1 : nnz);
    Lap = sparse(row, col, val, Nd, Nd, nnz);
    colidx = col;
end

function shift_ix = calc_fd_shift_pos(ix, Nx, BCx, FDn)
    shift_ix = zeros(2 * FDn + 1, 1) - 1;
    for r = -FDn : FDn
        shift_r = FDn + 1 + r;
        ixr = ix + r;
        if ((1 <= ixr) && (ixr <= Nx)), shift_ix(shift_r) = ixr;      end
        if ((ixr <  1) && (BCx == 0)),  shift_ix(shift_r) = ixr + Nx; end
        if ((ixr > Nx) && (BCx == 0)),  shift_ix(shift_r) = ixr - Nx; end
    end
end
