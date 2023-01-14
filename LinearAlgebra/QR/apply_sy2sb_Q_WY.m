function Z = apply_sy2sb_Q_WY(VB, bs, C)
% Apply the Q matrix from sy2sb to C (calculate Q * C)
% Input parameters:
%   VB : Size n * n, tril(VT, -bs-1) stores the Householder vectors in sy2sb
%   bs : Semi-bandwidth of the B matrix 
%   C  : Size n * k, matrix to be applied with Q
% Output parameter:
%   Z : Size n * k, == Q * C
    n = size(VB, 1);
    n1 = n - bs - 1;  % The last column in B that its last row is 0
    n2 = 1 + floor((n1 - 1) / bs) * bs;
    for j = n2 : -bs : 1
        % Columns in the current panel: [j, k]
        k = min(n1, j + bs - 1);
        curr_bs = k - j + 1;
        l = j + bs;
        % Extract Y from V and build W
        Y = tril(VB(l : n, j : k), -1) + eye(n-l+1, curr_bs);
        W = gen_W_from_Y(Y);
        % Q * C(l : n, :) = (I - W * Y') * C(l : n, :)
        YTC = Y' * C(l : n, :);
        C(l : n, :) = C(l : n, :) - W * YTC;
    end
    Z = C;
end