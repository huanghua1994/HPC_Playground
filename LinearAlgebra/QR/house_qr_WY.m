function VR = house_qr_WY(A, bs)
% Blocked Householder QR using WY representation
% Van Loan 4th edition Algorithm 5.2.2
% Input parameters:
%   A  : Size m * n, m >= n
%   bs : Block size, default is 8
% Output parameter:
%   VR : Same size as A, triu(VR(1:n, 1:n)) is R, tril(VR, -1) contains Householder vectors
    [m, n] = size(A);
    if (nargin < 2), bs = 8; end
    for j = 1 : bs : n
        % Columns in the current panel: [j, k]
        k = min(n, j + bs - 1);
        curr_bs = k - j + 1;
        % QR factorize A(j : m, j : k), in practice the results overwrite A(j : m, j : k)
        A(j : m, j : k) = house_qr(A(j : m, j : k));
        % Build W and Y from V
        % We only store and use the (j : m)-th non-zeros rows in Y and W
        Y = tril(A(j : m, j : k), -1) + eye(m-j+1, curr_bs);
        W = gen_W_from_Y(Y);
        % Update columns right to the current panel
        % Q_j * Q_{j+1} * ... * Q_k = eye(m) - W * Y'
        % A(j : m, k+1 : n) = Q_k * ... * Q_{j+1} * Q_j * A(j : m, k+1 : n)
        % A(j : m, k+1 : n) = (eye(m) - Y * W') * A(j : m, k+1 : n)
        WTA = W' * A(j : m, k+1 : n);
        A(j : m, k+1 : n) = A(j : m, k+1 : n) - Y * WTA;
    end
    VR = A;
end