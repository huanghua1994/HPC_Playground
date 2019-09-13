%% 1
stencil = zeros(5);
x = [1/12 -4/3 5 -4/3 1/12];
stencil(3, :) = x;
stencil(:, 3) = x';
test_stencil_2d_inv_precond( 300,  300, stencil, 1);
test_stencil_2d_inv_precond( 500,  500, stencil, 1);
test_stencil_2d_inv_precond(1000, 1000, stencil, 1);

%% 2
stencil = [-1 -1 -1; -1 8 -1; -1 -1 -1];
test_stencil_2d_inv_precond( 300,  300, stencil, 1);
test_stencil_2d_inv_precond( 500,  500, stencil, 1);
test_stencil_2d_inv_precond(1000, 1000, stencil, 1);

%% 3
stencil = zeros(3, 3, 3);
stencil(2, 2, 2) = 6;
stencil(1, 2, 2) = -1;
stencil(3, 2, 2) = -1;
stencil(2, 1, 2) = -1;
stencil(2, 3, 2) = -1;
stencil(2, 2, 1) = -1;
stencil(2, 2, 3) = -1;
test_stencil_3d_inv_precond( 80,  80,  80, stencil, 1);
test_stencil_3d_inv_precond(100, 100, 100, stencil, 1);
test_stencil_3d_inv_precond(120, 120, 120, stencil, 1);
