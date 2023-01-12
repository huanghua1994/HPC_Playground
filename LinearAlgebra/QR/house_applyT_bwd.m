function Y = house_applyT_bwd(V, X)
% Backward accumulation of the Householder matrices in reversed order
% Van Loan book 4th edition Section 5.1.6
% Input parameters:
%   V : Size m * n, its strict lower part contains the Householder vectors
%   X : Input vectors, size m * k
% Output parameter:
%   Y : Y = Qn * ... * Q2 * Q1 * X
    [m, n] = size(V);
    Y = X;
    for j = 1 : n
        v = [1; V(j+1 : m, j)];
        b = 2 / (norm(v).^2);
        % Y = Q_j * Y = (I - b * v * v') * Y
        t = v' * Y(j : m, :);
        Y(j : m, :) = Y(j : m, :) - (b .* v) * t;
    end
end