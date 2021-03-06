function test_stencil_2d_inv_precond(nrow, ncol, stencil, ext)
    A = stencil_2d_to_spmat(nrow, ncol, stencil);
    b = rand(size(A, 1), 1) - 0.5;
    [~, ~, ~, iter0] = pcg(A, b, 1e-6, 1000);
    
    M = stencil_2d_inv_precond(nrow, ncol, stencil, ext);
    prec1 = @(x)(M * x);
    [~, ~, ~, iter1] = pcg(A, b, 1e-6, 1000, prec1);
    
    L = ichol(A);
    [~, ~, ~, iter2] = pcg(A, b, 1e-6, 1000, L, L');
    fprintf('PCG no precond, inv_stencil, ichol iter = %d, %d, %d\n', iter0, iter1, iter2);
end