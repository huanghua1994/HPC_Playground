function VT = house_sb2st_v2(B, bs)
% Reduce a symmetric band matrix to symmetric tridiagonal matrix
% Reference: DOI 10.1109/SHPCC.1994.296622, bulge chasing algorithm
% Input parameters:
%   B  : Size n * n, symmetric matrix with bandwidth 2*bs+1
%   bs : Semi-bandwidth of B
% Output parameters:
%   VT : Size n * n, its tridiagonal part is T, tril(VT, -2) contains all
%        Householder vectors in sb2st 
    n = size(B, 1);
    % Sweep from 1st column to (n-2)-th column
    v_tmp = zeros(n, 1);
    for k = 1 : n-2
        j_list = [k, k+1 : bs : n-bs-1];
        for l = 1 : length(j_list)
            % Householder transform (k, l) in the paper
            % When transforming the j-th column, [r1, r2] are nonzeros, [r2+1, n] are zeros
            % [r1, r2]s in the same k do not overlap with each other
            j = j_list(l);
            if (l == 1)
                r1 = j + 1;
            else
                r1 = j + bs;
            end
            r2 = min(r1 + bs - 1, n);
            [v, b] = house_vec(B(r1 : r2, j));
            % The full Householder vector is [zeros(r1-1, 1); v; zeros(n-r2, 1)];
            % H' * B updates B(r1 : r2, j : r3), B * H updates B(j : r3, r1 : r2)
            if (l == 1)
                r3 = min(n, r1 + 2 * bs);
            else
                r3 = min(n, r1 + 3 * bs - 1);
            end
            t = v' * B(r1 : r2, j : r3);
            B(r1 : r2, j : r3) = B(r1 : r2, j : r3) - (b .* v) * t;
            t = B(j : r3, r1 : r2) * v;
            B(j : r3, r1 : r2) = B(j : r3, r1 : r2) - (b .* t) * v';
            v_tmp(r1 : r2) = v;
        end
        % v_tmp(k) is [zeros(1, k), 1, x, ..., x];
        B(k+2 : n, k) = v_tmp(k+2 : n);
    end
    VT = B;
end