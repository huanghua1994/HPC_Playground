function M = stencil_3d_submat_inv(nx, ny, nz, stencil, r)
% stencil is of size [2r+1, 2r+1, 2r+1]
    box_size = r * 2 + 1;
    box_vol  = box_size^3;
    cnt = 0;
    col = zeros(box_vol);
    val = zeros(box_vol);
    for iz = -r : r
    for iy = -r : r
    for ix = -r : r
        val_ixyz = stencil(ix + r + 1, iy + r + 1, iz + r + 1);
        if (abs(val_ixyz) > 0)
            cnt = cnt + 1;
            col(cnt) = iz * nx * ny + iy * nx + ix;
            val(cnt) = val_ixyz;
            if ((iz == 0) && (iy == 0) && (ix == 0))
                center_idx = cnt;
            end
        end
    end
    end
    end
    col = col(1 : cnt);
    val = val(1 : cnt);
    center_shift = -min(col) + 1;
    nnz_width = max(col) - min(col) + 1;
    subcols = zeros(nnz_width, cnt);
    for i = 1 : cnt
        subcols(col(i) + center_shift, center_idx) = val(i);
    end
    for j = 1 : cnt
        if (j == center_idx), continue; end
        shift = col(j) - col(center_idx);
        if (shift < 0)
            subcols(1 : nnz_width + shift, j) = subcols(1 - shift : nnz_width, center_idx);
        else
            subcols(1 + shift : nnz_width, j) = subcols(1 : nnz_width - shift, center_idx);
        end
    end
    subA = subcols(col + center_shift, :);
    subAinv = inv(subA);
    inv_stencil_nnz = subAinv(:, center_idx);
    inv_stencil = zeros(box_size);
    cnt = 0;
    for iz = -r : r
    for iy = -r : r
    for ix = -r : r
        val_ixyz = stencil(ix + r + 1, iy + r + 1, iz + r + 1);
        if (abs(val_ixyz) > 0)
            cnt = cnt + 1;
            inv_stencil(ix + r + 1, iy + r + 1, iz + r + 1) = inv_stencil_nnz(cnt);
        end
    end
    end
    end
    M = stencil_3d_to_spmat(nx, ny, nz, inv_stencil);
end