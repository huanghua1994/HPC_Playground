function [Q, T] = house_tridiag(A)
% Householder Tridiagonalization, Van Loan book 4th edition, Algorithm 8.3.1
% Output parameters:
%   Q : Product of Householder matrices
%   T : Tridiagonal matrix, A = Q^T * T * Q
    n = size(A, 1);
    Q = eye(n);
    for k = 1 : n-2
        [v, b] = house_vec(A(k+1 : n, k));
        p = b * A(k+1 : n, k+1 : n) * v;
        w = p - (0.5 * b * p' * v) * v;
        A(k+1, k) = norm(A(k+1 : n, k));
        A(k, k+1) = A(k+1, k);
        A(k+1 : n, k+1 : n) = A(k+1 : n, k+1 : n) - v * w' - w * v';
        A(k+2 : n, k) = 0;
        A(k, k+2 : n) = 0;

        % Apply Householder matrix, Q_k = Q_k-1 * H_k
        t = v' * Q(k+1 : n, :);
        Q(k+1 : n, :) = Q(k+1 : n, :) - b .* (v * t);
    end
    T = A;
end