n = 10;
A = rand(n);
A = A + A';

%%
fprintf('1-stage approach\n');
[Q, T] = house_tridiag(A);
[V1, D] = tridiag_eig_dc(T);
V = Q * V1;
e1 = norm(Q * T * Q' - A, 'fro') / norm(A, 'fro');
e2 = norm(V1 * D * V1' - T, 'fro') / norm(T, 'fro');
e3 = norm(V * D * V' - A, 'fro') / norm(A, 'fro');
fprintf('Householder tridiag:     ||Q  * T * Q^T  - A||_{fro} / ||A||_{fro} = %e\n', e1);
fprintf('Tridiag D&C eigensolve:  ||V1 * D * V1^T - T||_{fro} / ||T||_{fro} = %e\n', e2);
fprintf('Eigensolve (V = Q * V1): ||V  * D * V^T  - A||_{fro} / ||A||_{fro} = %e\n', e3);

%%
fprintf('2-stage approach\n');
bss = 2:4;
for bs = bss
    B_nnz_pattern = toeplitz([ones(1, 1+bs), zeros(1, n-bs-1)]);
    T_nnz_pattern = toeplitz([ones(1, 2), zeros(1, n-2)]);
    
    VB = house_sy2sb(A, bs);
    B = VB .* B_nnz_pattern;

    VT = house_sb2st_v2(B, bs);
    T = VT .* T_nnz_pattern;

    [V1, D] = tridiag_eig_dc(T);
    V2 = apply_sb2st_Q_WY(VT, bs, 4, V1);
    V = apply_sy2sb_Q_WY(VB, bs, V2);
    e = norm(V * D * V' - A, 'fro') / norm(A, 'fro');
    fprintf('bs = %d, ||V  * D * V^T  - A||_{fro} / ||A||_{fro} = %e\n', bs, e);
end