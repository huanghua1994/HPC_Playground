function x = AAR(A, x0, b, max_iter, res_tol)
    N = size(A, 1);
    m = 7;
    p = 6;
    omega = 0.1;
    beta  = 0.1;
    x = x0;
    b_l2norm = norm(b, 2);
    stop_res = b_l2norm * res_tol;
    X = zeros(N, m);
    F = zeros(N, m);
    inv_M = -1.0 / A(1, 1);
    for iter_cnt = 1 : max_iter
        r = b - A * x;
        f = r .* inv_M;
        if iter_cnt > 1
            k = mod(iter_cnt - 2, m) + 1;
            X(:, k) = x - x0;
            F(:, k) = f - f0;
        end
        x0 = x;
        f0 = f;
        
        if (mod(iter_cnt, p) == 0) && iter_cnt > 1
            FTF = F' * F;
            FTf = F' * f;
            XF  = (X + beta * F) * pinv(FTF) * FTf;
            x   = x0 + beta * f - XF;
        else
            x   = x0 + omega .* f;
        end
        
        r_l2norm = norm(r, 2);
        fprintf('%d: %e\n', iter_cnt, r_l2norm / b_l2norm);
        if (r_l2norm < stop_res), break; end
    end
end