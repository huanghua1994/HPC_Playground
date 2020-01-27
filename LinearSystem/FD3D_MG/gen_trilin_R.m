function [R, P] = gen_trilin_R(grid_sizes, BCs)
% Generate the 3D trilinear restriction matrix R
% Input parameters:
%   grid_sizes : Number of grid points on x, y, z direction
%   BCs        : Boundary condition on x, y, z direction, 0 - periodic, 1 - Dirichlet
% Output parameters:
%   R : 3D tri-linear restriction matrix R
    
    % Number of finite difference grid points
    Nx = grid_sizes(1);
    Ny = grid_sizes(2);
    Nz = grid_sizes(3);
    Nd = Nx * Ny * Nz;
    
    % Boundary conditions in each direction, 0 - periodic, 1 - Dirichlet
    BCx = BCs(1);
    BCy = BCs(2);
    BCz = BCs(3);
    
    M_Nx = floor(Nx / 2);
    M_Ny = floor(Ny / 2);
    M_Nz = floor(Nz / 2);
    M_Nd = M_Nx * M_Ny * M_Nz;
    R_row = zeros(27 * M_Nd, 1);
    R_col = zeros(27 * M_Nd, 1);
    R_val = zeros(27 * M_Nd, 1);
    R_nnz = 0;
    for iz = 2 : 2 : Nz
    for iy = 2 : 2 : Ny
    for ix = 2 : 2 : Nx
        M_iz  = iz / 2;
        M_iy  = iy / 2;
        M_ix  = ix / 2;
        M_idx = M_ix + (M_iy - 1) * M_Nx + (M_iz - 1) * M_Nx * M_Ny;
        
        [R_val_ixyz, R_col_ixyz] = gen_R_row_nnz(ix, iy, iz, Nx, Ny, Nz, BCx, BCy, BCz);
        nnz_ixyz = length(R_col_ixyz);
        R_nnz_range = (R_nnz + 1) : (R_nnz + nnz_ixyz);
        R_row(R_nnz_range) = M_idx;
        R_col(R_nnz_range) = R_col_ixyz;
        R_val(R_nnz_range) = R_val_ixyz;
        R_nnz = R_nnz + nnz_ixyz;
    end
    end
    end
    R_row = R_row(1 : R_nnz);
    R_col = R_col(1 : R_nnz);
    R_val = R_val(1 : R_nnz);
    R = sparse(R_row, R_col, R_val, M_Nd, Nd, R_nnz);
end

function [val, col] = gen_R_row_nnz(ix0, iy0, iz0, Nx, Ny, Nz, BCx, BCy, BCz)
    cnt = 0;
    val = zeros(27, 1);
    col = zeros(27, 1);
    for iz1 = iz0-1 : iz0+1
        iz = periodic_pos(iz1, Nz, BCz);
        if (iz == -1), continue; end
        for iy1 = iy0-1 : iy0+1
            iy = periodic_pos(iy1, Ny, BCy);
            if (iy == -1), continue; end
            for ix1 = ix0-1 : ix0+1
                ix = periodic_pos(ix1, Nx, BCx);
                if (ix == -1), continue; end
                dist = abs(iz1 - iz0) + abs(iy1 - iy0) + abs(ix1 - ix0);
                cnt = cnt + 1;
                col(cnt) = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny;
                val(cnt) = 0.125 * 2.^(-dist);
            end
        end
    end
    val = val(1 : cnt);
    col = col(1 : cnt);
end

function ix = periodic_pos(ix1, Nx, BCx)
    ix = -1;
    if (1 <= ix1 && ix1 <= Nx), ix = ix1;      end
    if (ix1 < 1  && BCx ==  0), ix = ix1 + Nx; end
    if (ix1 > Nx && BCx ==  0), ix = ix1 - Nx; end
end