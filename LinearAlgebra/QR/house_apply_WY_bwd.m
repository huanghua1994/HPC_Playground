function Z = house_apply_WY_bwd(V, X, bs)
% Backward accumulation of Householder matrices using WY representation
% Input parameters:
%   V  : Size m * n, its strict lower part contains the Householder vectors
%   X  : Input vectors, size m * k
%   bs : Block size, default is 8
% Output parameter:
%   Z : Z = Q1 * Q2 * ... * Qn * X
    [m, n] = size(V);
    if (nargin < 3), bs = 8; end
    n1 = min(m-1, n);  % The last column of a square V has no Householder vector
    Z = X;
    for k = n1 : -bs : 1
        % Columns in the current panel: [j, k]
        j = max(1, k - bs + 1);
        curr_bs = k - j + 1;
        % Build W and Y from V
        % We only store and use the (j : m)-th non-zeros rows in Y and W
        Y = tril(V(j : m, j : k));
        for jj = 1 : curr_bs
            Y(jj, jj) = 1;
        end
        W = gen_W_from_Y(Y);
        % Q_j * Q_{j+1} * ... * Q_k = eye(m) - W * Y'
        % Z = Q_j * Q_{j+1} * ... * Q_k * Z = (eye(m) - W * Y') * Z
        YTZ = Y' * Z(j : m, :);
        Z(j : m, :) = Z(j : m, :) - W * YTZ;
    end
end