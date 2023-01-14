function [Q, T] = house_sb2st(A, bs)
% Reduce a symmetric band matrix to symmetric tridiagonal matrix
% Reference: DOI 10.1109/SHPCC.1994.296622, bulge chasing algorithm
% Input parameters:
%   A  : Size n * n, symmetric matrix with bandwidth 2*bs+1
%   bs : Semi-bandwidth of A
% Output parameters:
%   Q : Size n * n, orthogonal matrix, Q * T * Q' == A
%   T : Size n * n, symmetric tridiagonal matrix
    n = size(A, 1);
    Q = eye(n);
    % Sweep from 1st column to (n-2)-th column
    for k = 1 : n-2
        j_list = [k, k+1 : bs : n-bs-1];
        j_len  = length(j_list);
        for j_idx = 1 : j_len
            % When transforming the j-th column, [r1, r2] are nonzeros, [r2+1, n] are zeros
            % [r1, r2]s in the same k do not overlap with each other
            j = j_list(j_idx);
            if (j_idx == 1)
                r1 = j + 1;
            else
                r1 = j + bs;
            end
            r2 = min(r1 + bs - 1, n);
            [v, b] = house_vec(A(r1 : r2, j));
            % The full Householder vector is [zeros(r1-1, 1); v; zeros(n-r2, 1)];
            % (H' * A) * H will first update A(r1 : r2, :), then update A(:, r1 : r2)
            H = eye(r2-r1+1) - (b * v) * v';
            A(r1 : r2, :) = H' * A(r1 : r2, :);
            A(:, r1 : r2) = A(:, r1 : r2) * H;
            % Not necessary, for debug and showing the change of A
            A(r1+1 : n, j) = 0;
            A(j, r1+1 : n) = 0;
            % Accumulate Q = Q * Q_k = Q - b * (Q * v) * v';
            % Each j reads and updates the same Q columns, and different j in
            % the same k reads and updates different Q columns (can be parallelized)
            t = Q(:, r1 : r2) * v;
            Q(:, r1 : r2) = Q(:, r1 : r2) - (b .* t) * v';
        end
    end
    T = A;
end