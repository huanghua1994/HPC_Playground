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
    coef_dxx = w2 ./ dx^2;
    coef_dyy = w2 ./ dy^2;
    coef_dzz = w2 ./ dz^2;

    max_nnz = Nd * (6 * FDn + 1);
    row = zeros(max_nnz, 1);
    col = zeros(max_nnz, 1);
    val = zeros(max_nnz, 1);
    rowptr = zeros(Nd + 1, 1);
    cnt = 0;
    for iz = 1 : Nz
    for iy = 1 : Ny
    for ix = 1 : Nx
        % (ix, iy, iz)
        cnt = cnt + 1;
        curr_row_idx = ix + (iy - 1) * Nx + (iz - 1) * Ny * Nx;
        row(cnt) = curr_row_idx;
        col(cnt) = curr_row_idx;
        val(cnt) = coef_dxx(1) + coef_dyy(1) + coef_dzz(1);
        rowptr(curr_row_idx) = cnt;
        for r = 1 : FDn
            % (ix + r, iy, iz)
            ixpr = ix + r;  flag = 0;
            if (ixpr <= Nx)
                flag = 1;
            elseif (BCx == 0)
                ixpr = ixpr - Nx;
                flag = 1;
            end
            if (flag == 1)
                cnt = cnt + 1;
                row(cnt) = curr_row_idx;
                col(cnt) = ixpr + (iy - 1) * Nx + (iz - 1) * Ny * Nx;
                val(cnt) = coef_dxx(r + 1);
            end
            % (ix - r, iy, iz)
            ixmr = ix - r;  flag = 0;
            if (ixmr >= 1)
                flag = 1;
            elseif (BCx == 0)
                ixmr = ixmr + Nx;
                flag = 1;
            end
            if (flag == 1)
                cnt = cnt + 1;
                row(cnt) = curr_row_idx;
                col(cnt) = ixmr + (iy - 1) * Nx + (iz - 1) * Ny * Nx;
                val(cnt) = coef_dxx(r + 1);
            end
            % (ix, iy + r, iz)
            iypr = iy + r;  flag = 0;
            if (iypr <= Ny)
                flag = 1;
            elseif (BCy == 0)
                iypr = iypr - Ny;
                flag = 1;
            end
            if (flag == 1)
                cnt = cnt + 1;
                row(cnt) = curr_row_idx;
                col(cnt) = ix + (iypr - 1) * Nx + (iz - 1) * Ny * Nx;
                val(cnt) = coef_dyy(r + 1);
            end
            % (ix, iy - r, iz)
            iymr = iy - r;  flag = 0;
            if (iymr >= 1)
                flag = 1;
            elseif (BCy == 0)
                iymr = iymr + Ny;
                flag = 1;
            end
            if (flag == 1)
                cnt = cnt + 1;
                row(cnt) = curr_row_idx;
                col(cnt) = ix + (iymr - 1) * Nx + (iz - 1) * Ny * Nx;
                val(cnt) = coef_dyy(r + 1);
            end
            % (ix, iy, iz + r)
            izpr = iz + r;  flag = 0;
            if (izpr <= Nz)
                flag = 1;
            elseif (BCz == 0)
                izpr = izpr - Nz;
                flag = 1;
            end
            if (flag == 1)
                cnt = cnt + 1;
                row(cnt) = curr_row_idx;
                col(cnt) = ix + (iy - 1) * Nx + (izpr - 1) * Ny * Nx;
                val(cnt) = coef_dzz(r + 1);
            end
            % (ix, iy, iz - r)
            izmr = iz - r;  flag = 0;
            if (izmr >= 1)
                flag = 1;
            elseif (BCz == 0)
                izmr = izmr + Nz;
                flag = 1;
            end
            if (flag == 1)
                cnt = cnt + 1;
                row(cnt) = curr_row_idx;
                col(cnt) = ix + (iy - 1) * Nx + (izmr - 1) * Ny * Nx;
                val(cnt) = coef_dzz(r + 1);
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
