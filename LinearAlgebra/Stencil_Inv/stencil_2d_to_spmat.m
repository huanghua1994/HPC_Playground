function spmat = stencil_2d_to_spmat(nx, ny, stencil)
    [nxs, nys] = size(stencil);
    maxnnz = nx * ny * nxs * nys;
    row = zeros(maxnnz, 1);
    col = zeros(maxnnz, 1);
    val = zeros(maxnnz, 1);
    semi_nxs = floor(nxs / 2);
    semi_nys = floor(nys / 2);
    cnt = 0;
    for x = 1 : nx
    for y = 1 : ny
        for sx = -semi_nxs : semi_nxs
        for sy = -semi_nys : semi_nys
            ix = x + sx;
            iy = y + sy;
            if (ix >= 1) && (ix <= nx) && (iy >= 1) && (iy <= ny)
                ival = stencil(sx + semi_nxs + 1, sy + semi_nys + 1);
                if (ival ~= 0)
                    cnt = cnt + 1;
                    row(cnt) = ( x - 1) * ny +  y;
                    col(cnt) = (ix - 1) * ny + iy;
                    val(cnt) = ival;
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