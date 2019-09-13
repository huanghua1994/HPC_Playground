function spmat = stencil_3d_to_spmat(nx, ny, nz, stencil)
    [nxs, nys, nzs] = size(stencil);
    maxnnz = nx * ny * nz * nxs * nys * nzs;
    row = zeros(maxnnz, 1);
    col = zeros(maxnnz, 1);
    val = zeros(maxnnz, 1);
    semi_nxs = floor(nxs / 2);
    semi_nys = floor(nys / 2);
    semi_nzs = floor(nzs / 2);
    cnt = 0;
    for x = 1 : nx
    for y = 1 : ny
    for z = 1 : nz
        for sx = -semi_nxs : semi_nxs
        for sy = -semi_nys : semi_nys
        for sz = -semi_nzs : semi_nzs
            ix = x + sx;
            iy = y + sy;
            iz = z + sz;
            if (ix >= 1) && (ix <= nx) && (iy >= 1) && (iy <= ny) && (iz >= 1) && (iz <= nz)
                ival = stencil(sx + semi_nxs + 1, sy + semi_nys + 1, sz + semi_nzs + 1);
                if (ival ~= 0)                
                    cnt = cnt + 1;
                    row(cnt) = ( x - 1) * ny * nz + ( y - 1) * nz +  z;
                    col(cnt) = (ix - 1) * ny * nz + (iy - 1) * nz + iz;
                    val(cnt) = ival;
                end
            end
        end
        end
        end
    end
    end
    end

    row = row(1 : cnt);
    col = col(1 : cnt);
    val = val(1 : cnt);
    spmat = sparse(row, col, val);
end