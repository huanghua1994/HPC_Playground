function [R, P, M] = gen_R_P_diag_RAP(grid_sizes, BCs, A_rowptr, A_col, A_val)
% Generate the 3D trilinear restriction matrix R, prolongation matrix P, 
% and the scaled diagonal of R * A * P for pre- and post-smoothing
% Input parameters:
%   grid_sizes : Number of grid points on x, y, z direction
%   BCs        : Boundary condition on x, y, z direction, 0 - periodic, 1 - Dirichlet
%   A_rowptr : A matrix in CSR format, row_ptr array
%   A_col    : A matrix in CSR format, col array
%   A_val    : A matrix in CSR format, val array
% Output parameters:
%   R : 3D trilinear restriction matrix R
%   P : 3D trilinear prolongation matrix P, P = 8 * R';
%   M : 0.75 ./ diag(R * A * P)
    
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
    M = zeros(M_Nd, 1);
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
        R_vec = zeros(Nd, 1);
        R_vec(R_col_ixyz) = R_val_ixyz;
        nnz_ixyz = length(R_col_ixyz);
        R_nnz_range = (R_nnz + 1) : (R_nnz + nnz_ixyz);
        R_row(R_nnz_range) = M_idx;
        R_col(R_nnz_range) = R_col_ixyz;
        R_val(R_nnz_range) = R_val_ixyz;
        R_nnz = R_nnz + nnz_ixyz;
        
        M_val = 0;
        for k = 1 : nnz_ixyz
            row = R_col_ixyz(k);
            res = 0;
            for j = A_rowptr(row) : (A_rowptr(row + 1) - 1)
                res = res + A_val(j) * R_vec(A_col(j));
            end
            M_val = M_val + res * R_val_ixyz(k);
        end
        M(M_idx) = 8 * M_val;
    end
    end
    end
    M = 0.75 ./ M;
    R_row = R_row(1 : R_nnz);
    R_col = R_col(1 : R_nnz);
    R_val = R_val(1 : R_nnz);
    R = sparse(R_row, R_col, R_val, M_Nd, Nd, R_nnz);
    P = 8 * R';
end

function [val, col] = gen_R_row_nnz(ix0, iy0, iz0, Nx, Ny, Nz, BCx, BCy, BCz)
    nnz = 0;
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
                nnz = nnz + 1;
                col(nnz) = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny;
                val(nnz) = 0.125 * 2.^(-dist);
            end
        end
    end
    val = val(1 : nnz);
    col = col(1 : nnz);
end

function ix = periodic_pos(ix1, Nx, BCx)
    ix = -1;
    if (1 <= ix1 && ix1 <= Nx), ix = ix1;      end
    if (ix1 < 1  && BCx ==  0), ix = ix1 + Nx; end
    if (ix1 > Nx && BCx ==  0), ix = ix1 - Nx; end
end