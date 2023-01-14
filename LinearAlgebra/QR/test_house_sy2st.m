ns = [100, 200, 300];
bss = [8, 15, 16, 30, 32];

fprintf('n, bs | sy2sb, sb2st, sy2st relerr\n');
for n = ns
    for bs = bss
        A = rand(n);
        A = A + A';

        [Q1, AB] = house_sy2sb(A, bs);
        AB_nnz_pattern = toeplitz([ones(1, 1+bs), zeros(1, n-bs-1)]);
        AB = AB .* AB_nnz_pattern;
        e1 = norm(Q1 * AB * Q1' - A, 'fro') / norm(A, 'fro');

        [Q2, T] = house_sb2st(AB, bs);
        T_nnz_pattern = toeplitz([ones(1, 3), zeros(1, n-3)]);
        T = T .* T_nnz_pattern;
        e2 = norm(Q2 * T * Q2' - AB, 'fro') / norm(AB, 'fro');
        
        Q = Q1 * Q2;
        e3 = norm(Q * T * Q' - A, 'fro') / norm(A, 'fro');
        fprintf('%d, %d\t| %5.2e, %5.2e, %5.2e\n', n, bs, e1, e2, e3);
    end
end