function [Q, T] = house_sb2st_v1(B, bs)
% Reduce a symmetric band matrix to symmetric tridiagonal matrix
% Reference: DOI 10.1109/SHPCC.1994.296622, bulge chasing algorithm
% Input parameters:
%   B  : Size n * n, symmetric matrix with bandwidth 2*bs+1
%   bs : Semi-bandwidth of B
% Output parameters:
%   Q : Size n * n, orthogonal matrix, Q * T * Q' == B
%   T : Size n * n, symmetric tridiagonal matrix
    n = size(B, 1);
    Q = eye(n);
    % Sweep from 1st column to (n-2)-th column
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
            % v can be stored in B(r1 : r2, k) for future use
            % Accumulate Q = Q * Q_k = Q - b * (Q * v) * v';
            % Each j reads and updates the same Q columns, and different j in
            % the same k reads and updates different Q columns (can be parallelized)
            if (k == 1)
                Q(r1 : r2, r1 : r2) = Q(r1 : r2, r1 : r2) - (b .* v) * v';
            else
                r4 = 2 + (l-1) * bs;
                r5 = min(n, r4 + k * bs);
                t = Q(r4 : r5, r1 : r2) * v;
                Q(r4 : r5, r1 : r2) = Q(r4 : r5, r1 : r2) - (b .* t) * v';
            end
        end
    end
    T = B;
end