function W = gen_W_from_Y(Y)
% Generate the WY matrix from Householder vectors
% Van Loan 4th edition Algorithm 5.1.2
% Y : Size m * r, m > r, each column is a Householder vector v
% W : Size m * r, generated W matrix
    [m, r] = size(Y);
    b = 2 / (norm(Y(:, 1)).^2);
    W = zeros(m, r);
    W(:, 1) = b * Y(:, 1);
    for j = 2 : r
        vj = Y(:, j);
        b = 2 / (norm(vj).^2);
        W(:, j) = b * vj - b * W(:, 1 : j-1) * (Y(:, 1 : j-1)' * vj);
    end
end