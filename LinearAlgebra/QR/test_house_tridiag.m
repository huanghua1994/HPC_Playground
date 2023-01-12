m = 10;
A = rand(m);
A = A + A';

%%
[Q, T] = house_tridiag(A);
[V1, D] = tridiag_eig_dc(T);
V = Q * V1;
e1 = norm(Q * T * Q' - A, 'fro') / norm(A, 'fro');
e2 = norm(V1 * D * V1' - T, 'fro') / norm(T, 'fro');
e3 = norm(V * D * V' - A, 'fro') / norm(A, 'fro');
fprintf('Householder tridiag:     ||Q  * T * Q^T  - A||_{fro} / ||A||_{fro} = %e\n', e1);
fprintf('Tridiag D&C eigensolve:  ||V1 * D * V1^T - T||_{fro} / ||T||_{fro} = %e\n', e2);
fprintf('Eigensolve (V = Q * V1): ||V  * D * V^T  - A||_{fro} / ||A||_{fro} = %e\n', e3);