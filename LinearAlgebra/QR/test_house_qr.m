m  = 200;
n  = 100;
k  = 80;
A  = rand(m, n) - 0.5;
IA = eye(m, n);
X  = rand(m, k) - 0.5;

%% Householder vector version
VR = house_qr(A);
R  = triu(VR(1 : n, 1 : n));
Q1 = house_apply_fwd(VR, IA);
Q2 = house_apply_bwd(VR, IA);
Y1 = house_apply_fwd(VR, X);
Y2 = house_apply_bwd(VR, X);
fprintf('Householder forward Q1\n'); 
e1 = norm(Q1 * R - A, 'fro') / norm(A, 'fro');
e2 = norm(Q1' * Q1 - eye(n), 'fro') / sqrt(n);
fprintf('  ||Q1   * R - A||_{fro} / ||A||_{fro} = %e\n', e1);
fprintf('  ||Q1^T * Q1 - I||_{fro} / ||I||_{fro} = %e\n', e2);
fprintf('Householder backward Q2\n'); 
e1 = norm(Q2 * R - A, 'fro') / norm(A, 'fro');
e2 = norm(Q2' * Q2 - eye(n), 'fro') / sqrt(n);
fprintf('  ||Q2   * R - A||_{fro} / ||A||_{fro} = %e\n', e1);
fprintf('  ||Q2^T * Q2 - I||_{fro} / ||I||_{fro} = %e\n', e2);
e3 = norm(Y1 - Y2, 'fro') / norm(Y1, 'fro');
fprintf('||Q1 * X - Q2 * X||_{fro} / ||Q1 * X||_{fro} = %e\n', e3);

%% Householder WY version
VR = house_qr_WY(A, 16);
R  = triu(VR(1 : n, 1 : n));
Q3 = house_apply_WY_fwd(VR, IA);
Q4 = house_apply_WY_bwd(VR, IA);
Y3 = house_apply_WY_fwd(VR, X);
Y4 = house_apply_WY_bwd(VR, X);
fprintf('Householder WY forward Q3\n'); 
e1 = norm(Q3 * R - A, 'fro') / norm(A, 'fro');
e2 = norm(Q3' * Q3 - eye(n), 'fro') / sqrt(n);
fprintf('  ||Q3   * R - A||_{fro} / ||A||_{fro} = %e\n', e1);
fprintf('  ||Q3^T * Q3 - I||_{fro} / ||I||_{fro} = %e\n', e2);
fprintf('Householder WY backward Q4\n'); 
e1 = norm(Q4 * R - A, 'fro') / norm(A, 'fro');
e2 = norm(Q4' * Q4 - eye(n), 'fro') / sqrt(n);
fprintf('  ||Q4   * R - A||_{fro} / ||A||_{fro} = %e\n', e1);
fprintf('  ||Q4^T * Q4 - I||_{fro} / ||I||_{fro} = %e\n', e2);
e3 = norm(Y3 - Y4, 'fro') / norm(Y3, 'fro');
fprintf('||Q3 * X - Q4 * X||_{fro} / ||Q3 * X||_{fro} = %e\n', e3);
