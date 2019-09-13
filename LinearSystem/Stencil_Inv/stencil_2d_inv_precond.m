function M = stencil_2d_inv_precond(nrow, ncol, stencil, ext)
    ext  = max(ext, 1);
    ss   = size(stencil, 1);
    ss1  = ss + 2 * ext;
    A    = stencil_2d_to_spmat(ss1, ss1, stencil);
    invA = inv(full(A));
    mid_idx = ceil((ss1 * ss1) / 2);
    inv_st  = reshape(invA(mid_idx, :), [ss1, ss1]);
    M = stencil_2d_to_spmat(nrow, ncol, inv_st);
end