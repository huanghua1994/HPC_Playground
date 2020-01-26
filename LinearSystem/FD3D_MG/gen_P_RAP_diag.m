function [P, M] = gen_P_RAP_diag(grid_sizes, BCs, A_rowptr, A_colidx, A_val)
% Generate the 3D trilinear prolongator matrix P and the scaled diagonal
% of R * A * P for pre- and post-smoothing
% Input parameters:
%   grid_sizes : Number of grid points on x, y, z direction
%   BCs        : Boundary condition on x, y, z direction, 0 - periodic, 1 - Dirichlet
%   A_rowptr   : A matrix in CSR format, row_ptr array
%   A_colidx   : A matrix in CSR format, col_idx array
%   A_val      : A matrix in CSR format, val array
% Output parameters:
%   P : 3D trilinear prolongator matrix P
%   M : 0.75 ./ diag(R * A * P), where R = 0.125 * P'
    
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
    P_row = zeros(27 * M_Nd, 1);
    P_col = zeros(27 * M_Nd, 1);
    P_val = zeros(27 * M_Nd, 1);
    P_nnz = 0;
    for iz = 2 : 2 : Nz
    for iy = 2 : 2 : Ny
    for ix = 2 : 2 : Nx
        M_iz  = iz / 2;
        M_iy  = iy / 2;
        M_ix  = ix / 2;
        M_idx = M_ix + (M_iy - 1) * M_Nx + (M_iz - 1) * M_Nx * M_Ny;
        
        [P_val_ixyz, P_row_ixyz] = gen_P_stencil_vector(ix, iy, iz, Nx, Ny, Nz, BCx, BCy, BCz);
        P_vec = zeros(Nd, 1);
        P_vec(P_row_ixyz) = P_val_ixyz;
        nnz_ixyz = length(P_row_ixyz);
        P_nnz_range = (P_nnz + 1) : (P_nnz + nnz_ixyz);
        P_col(P_nnz_range) = M_idx;
        P_row(P_nnz_range) = P_row_ixyz;
        P_val(P_nnz_range) = P_val_ixyz;
        P_nnz = P_nnz + nnz_ixyz;
        
        M_val = 0;
        for k = 1 : nnz_ixyz
            row = P_row_ixyz(k);
            res = 0;
            for j = A_rowptr(row) : (A_rowptr(row + 1) - 1)
                res = res + A_val(j) * P_vec(A_colidx(j));
            end
            M_val = M_val + res * P_val_ixyz(k);
        end
        M(M_idx) = 0.125 * M_val;
    end
    end
    end
    M = 0.75 ./ M;
    P_row = P_row(1 : P_nnz);
    P_col = P_col(1 : P_nnz);
    P_val = P_val(1 : P_nnz);
    P = sparse(P_row, P_col, P_val, Nd, M_Nd, P_nnz);
end

function [val, idx] = gen_P_stencil_vector(ix0, iy0, iz0, Nx, Ny, Nz, BCx, BCy, BCz)
    nnz = 0;
    val = zeros(27, 1);
    idx = zeros(27, 1);
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
                idx(nnz) = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny;
                val(nnz) = 2.^(-dist);
            end
        end
    end
    val = val(1 : nnz);
    idx = idx(1 : nnz);
end

function ix = periodic_pos(ix1, Nx, BCx)
    ix = -1;
    if (1 <= ix1 && ix1 <= Nx), ix = ix1;      end
    if (ix1 < 0  && BCx ==  0), ix = ix1 + Nx; end
    if (ix1 > Nx && BCx ==  0), ix = ix1 - Nx; end
end