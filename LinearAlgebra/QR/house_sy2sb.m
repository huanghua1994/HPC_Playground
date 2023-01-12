function [Q, AB] = house_sy2sb(A, bs)
% Reduces a real symmetric matrix A to real symmetric band-diagonal
% form AB using blocked Householder transforms
% Reference: 10.1145/44128.44130, Section 3
% Input parameter:
%   A  : Size n * n, symmetric matrix
%   bs : Semi bandwidth of T, >= 2, default is 8
% Output parameters:
%   Q : Size n * n, orthogonal matrix, A = Q * AB * Q^T
%   T : Size n * n, band-diagonal matrix with bandwidth 2*bs+1
    n = size(A, 1);
    Q = eye(n);
    if (nargin < 2), bs = 8; end
    n1 = n - bs - 1;  % The last column in AB that its last row is 0
    for j = 1 : bs : n1
        % Columns in the current panel: [j, k]
        k = min(n1, j + bs - 1);
        curr_bs = k - j + 1;
        % Note: the first row of A in the panel is l = j + bs, not k + 1
        % QR factorize M = A(l : n, j : k), size(M, 1) >= size(M, 2) always holds
        % In practice the results overwrite A(l : n, j : k)
        l = j + bs;
        VR = house_qr(A(l : n, j : k));
        R = triu(VR(1 : curr_bs, :));
        A(l : n, j : k) = 0;
        A(j : k, l : n) = 0;
        A(l : l+curr_bs-1, j : k) = R;
        A(j : k, l : l+curr_bs-1) = R';
        if (k < l-1)
            % Last block, where l > k, need to handle columns [k+1, l-1]
            A(l : n, k+1 : l-1) = house_applyT_bwd(VR, A(l : n, k+1 : l-1));
            A(k+1 : l-1, l : n) = A(l : n, k+1 : l-1)';
        end
        % Build W and Y from V
        Y = tril(VR(:, 1 : curr_bs), -1) + eye(n-l+1, curr_bs);
        W = gen_W_from_Y(Y);
        % Q_k = eye(n) - W * Y', apply Q_k' * A * Q_k
        S = A(l : n, l : n) * W;
        V = W' * S;
        T = S - 0.5 * Y * V;
        YTT = Y * T';
        A(l : n, l : n) = A(l : n, l : n) - YTT - YTT';
        % Accumulate Q = Q * Q_k = Q - Q * W * Y'
        QW = Q(:, l : n) * W;
        Q(:, l : n) = Q(:, l : n) - QW * Y';
    end
    AB = A;
end