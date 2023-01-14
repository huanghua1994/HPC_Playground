function Y = house_apply_fwd(V, X)
% Forward accumulation of Householder matrices
% Van Loan book 4th edition Section 5.1.6
% Input parameters:
%   V : Size m * n, its strict lower part contains the Householder vectors
%   X : Input vectors, size m * k
% Output parameter:
%   Y : Y = Q1 * Q2 * ... * Qn * X
    [m, n] = size(V);
    Q = eye(m);
    for j = 1 : n
        v = [1; V(j+1 : m, j)];
        b = 2 / (norm(v).^2);
        % Q = Q * Q_j = Q * (I - b * v * v')
        t = Q(:, j : m) * v;
        Q(:, j : m) = Q(:, j : m) - (b .* t) * v';
    end
    Y = Q * X;
end