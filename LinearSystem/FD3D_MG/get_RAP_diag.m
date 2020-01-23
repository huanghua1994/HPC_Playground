function M = get_RAP_diag(grid_sizes, BCs, A_rowptr, A_colidx, A_val)
    
    % Number of finite difference grid points
    Nx = grid_sizes(1);
    Ny = grid_sizes(2);
    Nz = grid_sizes(3);
    Nd = Nx * Ny * Nz;
    
    % Boundary conditions in each direction, 0 - periodic, 1 - Dirichlet
    BCx = BCs(1);
    BCy = BCs(2);
    BCz = BCs(3);
    
    M_Nz = length(2 : 2 : Nz);
    M_Ny = length(2 : 2 : Ny);
    M_Nx = length(2 : 2 : Nx);
    M = zeros(M_Nx * M_Ny * M_Nz, 1);
    for iz = 2 : 2 : Nz
    for iy = 2 : 2 : Ny
    for ix = 2 : 2 : Nx
        M_iz  = iz / 2;
        M_iy  = iy / 2;
        M_ix  = ix / 2;
        M_idx = M_ix + (M_iy - 1) * M_Nx + (M_iz - 1) * M_Nx * M_Ny;
        
        [PR_val, PR_idx] = gen_PR_stencil_vector(ix, iy, iz, Nx, Ny, Nz, BCx, BCy, BCz);
        PR_vec = zeros(Nd, 1);
        PR_vec(PR_idx) = PR_val;
        
        M_val = 0;
        for k = 1 : length(PR_idx)
            row = PR_idx(k);
            res = 0;
            for j = A_rowptr(row) : A_rowptr(row+1)-1
                res = res + A_val(j) * PR_vec(A_colidx(j));
            end
            M_val = M_val + res * PR_val(k);
        end
        M(M_idx) = 0.125 * M_val;
    end
    end
    end
    M = 0.75 ./ M;
end

function [val, idx] = gen_PR_stencil_vector(ix0, iy0, iz0, Nx, Ny, Nz, BCx, BCy, BCz)
    cnt = 0;
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
                cnt = cnt + 1;
                idx(cnt) = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny;
                val(cnt) = 2.^(-dist);
            end
        end
    end
    val = val(1 : cnt);
    idx = idx(1 : cnt);
end

function ix = periodic_pos(ix1, Nx, BCx)
    ix = ix1;
    if (ix1 < 1)
        if (BCx == 1)
            ix = -1;
        else 
            ix = ix + Nx;
        end
    end
    if (ix1 > Nx)
        if (BCx == 1)
            ix = -1;
        else 
            ix = ix - Nx;
        end
    end
end