function [Q, T] = house_tridiag(A)
% Householder Tridiagonalization, Van Loan book 4th edition, Algorithm 8.3.1
% Output parameters:
%   Q : Product of Householder matrices
%   T : Tridiagonal matrix, A = Q * T * Q^T
    n = size(A, 1);
    Q = eye(n);
    for k = 1 : n-2
        [v, b] = house_vec(A(k+1 : n, k));
        % Q_k = eye(m) - b * v * v', apply Q_k' * A * Q_k
        p = b * A(k+1 : n, k+1 : n) * v;
        w = p - (0.5 * b * p' * v) * v;
        A(k+1, k) = norm(A(k+1 : n, k));
        A(k, k+1) = A(k+1, k);
        vwT = v * w';
        A(k+1 : n, k+1 : n) = A(k+1 : n, k+1 : n) - vwT - vwT';
        % v(2 : end) can be stored in A(k+2 : n, k) 
        A(k+2 : n, k) = 0;
        A(k, k+2 : n) = 0;
        % Accumulate Q = Q * Q_k = Q - b * (Q * v) * v';
        t = Q(:, k+1 : n) * v;
        Q(:, k+1 : n) = Q(:, k+1 : n) - (b .* t) * v';
    end
    T = A;
end