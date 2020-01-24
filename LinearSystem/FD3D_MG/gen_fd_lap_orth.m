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
    for k = 1 : FDn
        w2(k+1) = (2*(-1)^(k+1))*(factorial(FDn)^2) / ...
                    (k * k * factorial(FDn-k) * factorial(FDn+k));
        w2(1) = w2(1)-2*(1/(k*k));
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

    max_nnz = Nd * (6 * FDn + 1);
    row = zeros(max_nnz, 1);
    col = zeros(max_nnz, 1);
    val = zeros(max_nnz, 1);
    rowptr = zeros(Nd + 1, 1);
    cnt = 0;
    for iz = 1 : Nz
        shift_iz = calc_fd_shift_pos(iz, Nz, BCz, FDn);
        for iy = 1 : Ny
            shift_iy = calc_fd_shift_pos(iy, Ny, BCy, FDn);
            for ix = 1 : Nx
                shift_ix = calc_fd_shift_pos(ix, Nx, BCx, FDn);
                % (ix, iy, iz)
                curr_row_idx = ix + (iy - 1) * Nx + (iz - 1) * Ny * Nx;
                cnt = cnt + 1;
                row(cnt) = curr_row_idx;
                col(cnt) = curr_row_idx;
                val(cnt) = coef_dxx(FDn+1) + coef_dyy(FDn+1) + coef_dzz(FDn+1);
                rowptr(curr_row_idx) = cnt;
                % (ix +- r, iy, iz)
                % (ix, iy +- r, iz)
                % (ix, iy, iz +- r)
                for r = -FDn : FDn
                    if (r == 0), continue; end
                    shift_r = FDn + 1 + r;
                    ix1 = shift_ix(shift_r);
                    iy1 = shift_iy(shift_r);
                    iz1 = shift_iz(shift_r);
                    if (ix1 ~= -1)
                        cnt = cnt + 1;
                        row(cnt) = curr_row_idx;
                        col(cnt) = ix1 + (iy - 1) * Nx + (iz - 1) * Ny * Nx;
                        val(cnt) = coef_dxx(shift_r);
                    end
                    if (iy1 ~= -1)
                        cnt = cnt + 1;
                        row(cnt) = curr_row_idx;
                        col(cnt) = ix + (iy1 - 1) * Nx + (iz - 1) * Ny * Nx;
                        val(cnt) = coef_dyy(shift_r);
                    end
                    if (iz1 ~= -1)
                        cnt = cnt + 1;
                        row(cnt) = curr_row_idx;
                        col(cnt) = ix + (iy - 1) * Nx + (iz1 - 1) * Ny * Nx;
                        val(cnt) = coef_dzz(shift_r);
                    end
                end
            end
        end
    end
    rowptr(Nd + 1) = cnt + 1;
    
    row = row(1 : cnt);
    col = col(1 : cnt);
    val = val(1 : cnt);
    Lap = sparse(row, col, val, Nd, Nd, cnt);
    colidx = col;
end

function ix_shift = calc_fd_shift_pos(ix0, Nx, BCx, FDn)
    ix_shift = zeros(2 * FDn + 1, 1) - 1;
    for r = -FDn : FDn
        shift_r = FDn + 1 + r;
        ix0pr = ix0 + r;
        if ((1 <= ix0pr) && (ix0pr <= Nx))
            ix_shift(shift_r) = ix0pr;
        end
        if ((ix0pr <  1) && (BCx == 0))
            ix0pr = ix0pr + Nx;
            ix_shift(shift_r) = ix0pr;
        end
        if ((ix0pr > Nx) && (BCx == 0))
            ix0pr = ix0pr - Nx;
            ix_shift(shift_r) = ix0pr;
        end
    end
end
