function [V, D] = tridiag_eig_dc(T)
% Divide and conquer method, see Van Load 4th edition Section 8.4.3, 8.4.4
% T is a symmetric tridiagonal matrix, V * D * V' = T
    n = size(T, 1);
    if (n <= 4)
        [V, D] = eig(T);
        return;
    end
    % T = blkdiag(T1, T2) + rho * v * v';
    % v = [zeros(m-1, 1); 1; 1; zeros(n-m-1, 1)];
    m = floor(n / 2);
    rho = T(m+1, m);  % == beta_m
    T1 = T(1 : m, 1 : m);
    T1(m, m) = T1(m, m) - rho;
    [Q1, D1] = tridiag_eig_dc(T1);
    T2 = T(m+1 : n, m+1 : n);
    T2(1, 1) = T2(1, 1) - rho;
    [Q2, D2] = tridiag_eig_dc(T2);
    % U = blkdiag(Q1, Q2); D = blkdiag(D1, D2);
    % U' * T * U = U' * (blkdiag(T1, T2) + rho * v * v') * U = D + rho * z * z'; 
    % z = U' * v = [Q1(m, :), Q2(1, :)]';
    % Solve [V1, L1] = eig(D + rho * z * z')
    z  = [Q1(m, :), Q2(1, :)]';
    d0 = [diag(D1); diag(D2)];
    d  = sort(d0, 'descend');
    L1 = zeros(n);
    V1 = zeros(n);
    % Theorem 8.4.3
    % (a) l_i are the n zeros of 1 + rho * z' * inv(D - l * I) * z
    f = @(lambda)(1 + rho * sum( z .* z ./ (d0 - lambda)));
    for i = 1 : n
        if (rho > 0)
            % (b-1) l_1 > d_1 > l_2 > d_2 > ... > l_n > d_n
            d_i = d(i) + 10 * eps;
            if (i == 1)
                h = 1;
                d_im1 = d(1) + h;
                f_d1 = f(d(1) + 10 * eps);
                while (f(d_im1) * f_d1 > 0)
                    h = h * 2;
                    d_im1 = d(1) + h;
                end
            else
                d_im1 = d(i - 1) - 10 * eps;
            end
            l_i_range = [d_i, d_im1];
        else
            % (b-2) d_1 > l_1 > d_2 > l_2 > ... > d_n > l_n
            d_i = d(i) - 10 * eps;
            if (i == n)
                h = 1;
                d_ip1 = d(n) - h;
                f_dn = f(d(n) - 10 * eps);
                while (f(d_ip1) * f_dn > 0)
                    h = h * 2;
                    d_ip1 = d(n) - h;
                end
            else
                d_ip1 = d(i + 1) + 10 * eps;
            end
            l_i_range = [d_ip1, d_i];
        end
        l_i = fzero(f, l_i_range);
        L1(i, i) = l_i;
        % (c) Eigenvector v_i is a multiple of inv(D - l_i * I) * z
        v_i = z ./ (d0 - l_i);
        V1(:, i) = v_i ./ norm(v_i);
    end
    % U' * T * U = V1 * L1 * V1';
    % T = (U * V1) * L1 * (V1' * U');
    V = [Q1 * V1(1 : m, :); Q2 * V1(m+1 : n, :)];
    D = L1;
end
