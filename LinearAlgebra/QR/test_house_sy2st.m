ns = [256, 320];
bss = [2:8, 10:3:32];

fprintf('n, bs | sy2sb, sb2st, sy2st relerr\n');
for n = ns
    for bs = bss
        A = rand(n);
        A = A + A';

        VB = house_sy2sb(A, bs);
        B_nnz_pattern = toeplitz([ones(1, 1+bs), zeros(1, n-bs-1)]);
        B = VB .* B_nnz_pattern;
        Q1 = apply_sy2sb_Q_WY(VB, bs, eye(n));
        e1 = norm(Q1 * B * Q1' - A, 'fro') / norm(A, 'fro');

        T_nnz_pattern = toeplitz([ones(1, 2), zeros(1, n-2)]);
        VT = house_sb2st_v2(B, bs);
        Q2 = apply_sb2st_Q_WY(VT, bs, 8, eye(n));
        T = VT .* T_nnz_pattern;
        e2 = norm(Q2 * T * Q2' - B, 'fro') / norm(B, 'fro');
        
        Q = apply_sy2sb_Q_WY(VB, bs, Q2);
        e3 = norm(Q * T * Q' - A, 'fro') / norm(A, 'fro');
        fprintf('%d, %d\t| %5.2e, %5.2e, %5.2e\n', n, bs, e1, e2, e3);
    end
end