function VR = house_qr(A)
% Householder QR, Van Loan book 4th edition Algorithm 5.2.1
% Output parameters:
%   VR : Same size as A, triu(VR(1:n, 1:n)) is R, tril(VR, -1) contains Householder vectors
    [m, n] = size(A);
    for j = 1 : n
        if (j == m), break; end
        [v, b] = house_vec(A(j : m, j));
        t = v' * A(j : m, j : n);
        A(j : m, j : n) = A(j : m, j : n) - b .* (v * t);
        A(j+1 : m, j) = v(2 : m-j+1);
    end
    VR = A;
end