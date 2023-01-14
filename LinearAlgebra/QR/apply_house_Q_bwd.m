function Y = apply_house_Q_bwd(V, X)
% Backward accumulation of Householder matrices
% Van Loan book 4th edition Section 5.1.6
% Input parameters:
%   V : Size m * n, its strict lower part contains the Householder vectors
%   X : Input vectors, size m * k
% Output parameter:
%   Y : Y = Q1 * Q2 * ... * Qn * X
    [m, n] = size(V);
    Y = X;
    for j = n : -1 : 1
        v = [1; V(j+1 : m, j)];
        b = 2 / (norm(v).^2);
        % Y = Q_j * Y = (I - b * v * v') * Y
        t = v' * Y(j : m, :);
        Y(j : m, :) = Y(j : m, :) - (b .* v) * t;
    end
end