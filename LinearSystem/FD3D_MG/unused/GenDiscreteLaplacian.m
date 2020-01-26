function DL = GenDiscreteLaplacian(cellDims,gridsizes,latVecs,BCs,FDn)
% @brief    GenDiscreteLaplacian(cellDims,gridsizes,latVecs,BCs,FDn) 
%           discretizes the Laplacian operator in a parallelepiped domain 
%           (with a mixture of homogeneous Dirichlet boundary conditions 
%           and periodic boundary conditions) and returns the matrix DL.
%
% @param cellDims   The lengths of the domain in the lattice vector
%                   directions. 
%                       cellDims = [Lx,Ly,Lz], 1x3 vector.
% @param gridsizes  The numbers of finite-difference grids in the
%                   lattice vector directions. 
%                       gridsizes = [Nx,Ny,Nz], 1x3 vector.
% @param latVecs    The lattice vectors. 
%                       latVecs = [u;v;w], 3x3 matrix.
%                   where u, v, w are 1x3 unit vectors denoting the
%                   lattice vectors. For orthogonal systems, latVecs
%                   is just the 3x3 identity matrix.
% @param BCs        Boundary conditions in the lattice vector
%                   directions, 0 - periodic, 1 - dirichlet.
%                       BCs = [BCx,BCy,BCz], 1x3 vector.
% @param FDn        Half of the order of the finite difference scheme.
%
% @authors  Qimen Xu <qimenxu@gatech.edu>
%           Abhiraj Sharma <asharma424@gatech.edu>
%           Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
% @copyright (c) 2019 Material Physics & Mechanics Group, Georgia Tech
%

S = initialize(cellDims,gridsizes,latVecs,BCs,FDn);

% Calculate discrete laplacian (1D) and discrete gradient indices' values
S = lapIndicesValues_1d(S);
S = gradIndicesValues(S);

% Calculate discrete laplacian
[DL11,DL22,DL33,DG1,DG2,DG3] = blochLaplacian_1d(S,[0 0 0]);

DL = S.lapc_T(1,1) * kron(speye(S.Nz),kron(speye(S.Ny),DL11)) + ...
     S.lapc_T(2,2) * kron(speye(S.Nz),kron(DL22,speye(S.Nx))) + ...
     S.lapc_T(3,3) * kron(DL33,kron(speye(S.Ny),speye(S.Nx)));
if S.cell_typ == 2 % for non-orthogonal cell
	fprintf(2,'Creating mixed derivatives ...\n');
	% mixed derivatives
	MDL = S.lapc_T(1,2) * kron(speye(S.Nz),kron(DG2,DG1)) + ...
		  S.lapc_T(2,3) * kron(DG3,kron(DG2,speye(S.Nx))) + ...
		  S.lapc_T(1,3) * kron(DG3,kron(speye(S.Ny),DG1));
	DL = DL + MDL;
end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S = initialize(cellDims,gridsizes,latVecs,BCs,FDn)
% PARSING INPUT ARGUMENTS

% cell lengths in the lattice vector directions
Lx = cellDims(1);
Ly = cellDims(2);
Lz = cellDims(3);

% boundary conditions in each direction, 0 - periodic, 1 - dirichlet
BCx = BCs(1);
BCy = BCs(2);
BCz = BCs(3);

% number of finite difference grid points
Nx = gridsizes(1);
Ny = gridsizes(2);
Nz = gridsizes(3);
Nd = Nx * Ny * Nz; % total number of FD grid points

% number of finite difference intervals
Nintv_x = Nx - BCx;
Nintv_y = Ny - BCy;
Nintv_z = Nz - BCz;

% SETTING UP NECESSARY VARIABLES
% mesh spacing
dx = Lx/Nintv_x;
dy = Ly/Nintv_y;
dz = Lz/Nintv_z;


% Finite difference weights for the second derivative
w2 = zeros(1,FDn+1) ; 
for k=1:FDn
    w2(k+1) = (2*(-1)^(k+1))*(factorial(FDn)^2)/...
                    (k*k*factorial(FDn-k)*factorial(FDn+k));
    w2(1) = w2(1)-2*(1/(k*k));
end

% Finite difference weights for the first derivative
w1 = zeros(1,FDn) ; 
for k=1:FDn
    w1(k+1) = ((-1)^(k+1))*(factorial(FDn)^2)/...
                (k*factorial(FDn-k)*factorial(FDn+k));
end

% Laplacian transformation matrices
% first make sure all three lattice vectors are normalized
latVecs(1,:) = latVecs(1,:)/norm(latVecs(1,:)); 
latVecs(2,:) = latVecs(2,:)/norm(latVecs(2,:));
latVecs(3,:) = latVecs(3,:)/norm(latVecs(3,:));

% check if domain is orthogonal
tol = 1e-14;
isOrth = max(max(abs(latVecs - eye(3)))) < tol;
if isOrth
	cell_typ = 1;
else
	cell_typ = 2;
end
% create transformation matrix
grad_T = inv(latVecs');
lapc_T = grad_T * grad_T';
lapc_T(1,2) = 2*lapc_T(1,2); 
lapc_T(2,3) = 2*lapc_T(2,3);
lapc_T(1,3) = 2*lapc_T(1,3);

% save in structure S
S.Lx = Lx; S.Ly = Ly; S.Lz = Lz;
S.BCx = BCx; S.BCy = BCy; S.BCz = BCz;
S.dx = dx; S.dy = dy; S.dz = dz;
S.Nintv_x = Nintv_x; S.Nintv_y = Nintv_y; S.Nintv_z = Nintv_z;
S.Nx = Nx; S.Ny = Ny; S.Nz = Nz; S.N = Nd;
S.FDn = FDn;
S.w1 = w1;
S.w2 = w2;
S.latVecs = latVecs;
S.cell_typ = cell_typ;
S.lapc_T = lapc_T;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DL11,DL22,DL33,DG1,DG2,DG3] = blochLaplacian_1d(S,kptvec)
% @ brief    Calculates each component of laplacian in 1D
% @ authors
%         Abhiraj Sharma <asharma424@gatech.edu>
%         Qimen Xu <qimenxu@gatech.edu>
%         Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
%
% @param kptvec      k-point vector for the current Block diagonalized problem
% @param DLii        Discrete laplacian component in 1D along ith direction
% @param DGi         Discrete gradient component in 1D along ith direction
%
% @copyright (c) 2019 Material Physics & Mechanics Group, Georgia Tech
%============================================================================
Nx = S.Nx; Ny = S.Ny; Nz = S.Nz;

% Phase factors
if kptvec(1) == 0
	x_phasefac_l = 1.0;
	x_phasefac_r = 1.0;
else
	x_phasefac_l = exp(-1i*kptvec(1)*S.L1);
	x_phasefac_r = exp(1i*kptvec(1)*S.L1);
end

if kptvec(2) == 0
	y_phasefac_l = 1.0;
	y_phasefac_r = 1.0;
else
	y_phasefac_l = exp(-1i*kptvec(2)*S.L2);
	y_phasefac_r = exp(1i*kptvec(2)*S.L2);
end

if kptvec(3) == 0
	z_phasefac_l = 1.0;
	z_phasefac_r = 1.0;
else
	z_phasefac_l = exp(-1i*kptvec(3)*S.L3);
	z_phasefac_r = exp(1i*kptvec(3)*S.L3);
end


% D_xx laplacian in 1D
%-----------------------
V = S.V_11;

if S.BCx == 0
	V(S.isOutl_11) = V(S.isOutl_11) * x_phasefac_l;
	V(S.isOutr_11) = V(S.isOutr_11) * x_phasefac_r;
end

% Create discretized Laplacian
DL11 = sparse(S.I_11,S.II_11,V,Nx,Nx);

% D_yy laplacian in 1D
%-----------------------
V = S.V_22;

if S.BCy == 0
	V(S.isOutl_22) = V(S.isOutl_22) * y_phasefac_l;
	V(S.isOutr_22) = V(S.isOutr_22) * y_phasefac_r;  
end

% Create discretized Laplacian
DL22 = sparse(S.I_22,S.II_22,V,Ny,Ny);

% D_zz laplacian in 1D
%-----------------------

V = S.V_33;

if S.BCz == 0
	V(S.isOutl_33) = V(S.isOutl_33) * z_phasefac_l;
	V(S.isOutr_33) = V(S.isOutr_33) * z_phasefac_r;  
end

% Create discretized Laplacian
DL33 = sparse(S.I_33,S.II_33,V,Nz,Nz);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DG1 = zeros(Nx,Nx);
DG2 = zeros(Ny,Ny);
DG3 = zeros(Nz,Nz);

if S.cell_typ > 1
	% x-direction
	%-------------

	V = S.V_1;

	if S.BCx == 0
		V(S.isOutl_1) = V(S.isOutl_1) * x_phasefac_l;
		V(S.isOutr_1) = V(S.isOutr_1) * x_phasefac_r;  
	end

	% Create discretized Laplacian
	DG1 = sparse(S.I_1,S.II_1,V,Nx,Nx);

	% y-direction
	%-------------

	V = S.V_2;

	if S.BCy == 0
		V(S.isOutl_2) = V(S.isOutl_2) * y_phasefac_l;
		V(S.isOutr_2) = V(S.isOutr_2) * y_phasefac_r;  
	end

	% Create discretized Laplacian
	DG2 = sparse(S.I_2,S.II_2,V,Ny,Ny);

	% z-direction
	%-------------

	V = S.V_3;

	if S.BCz == 0
		V(S.isOutl_3) = V(S.isOutl_3) * z_phasefac_l;
		V(S.isOutr_3) = V(S.isOutr_3) * z_phasefac_r;  
	end

	% Create discretized Laplacian
	DG3 = sparse(S.I_3,S.II_3,V,Nz,Nz);
	
end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S = lapIndicesValues_1d(S)
% @ brief    Calculates laplacian and gradient (in 1D) indices' values without Bloch factor
% @ authors
%         Abhiraj Sharma <asharma424@gatech.edu>
%         Qimen Xu <qimenxu@gatech.edu>
%         Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
%
% @copyright (c) 2019 Material Physics & Mechanics Group, Georgia Tech
%============================================================================

Nx = S.Nx; Ny = S.Ny; Nz = S.Nz;
FDn = S.FDn;
w1 = S.w1;
w2 = S.w2;
dx = S.dx;
dy = S.dy;
dz = S.dz;

% D_xx laplacian in 1D
%-----------------------

% Initial number of non-zeros: including ghost nodes
nnzCount = (2 * FDn + 1) * Nx;

% Row and column indices and the corresponding non-zero values
% used to generate sparse matrix DL11 s.t. DL11(I(k),II(k)) = V(k)
I = zeros(nnzCount,1);
V = zeros(nnzCount,1);
II = zeros(nnzCount,1);
rowCount = 1;
count = 1;
coef_dxx = 1/dx^2;

% Find non-zero entries that use forward difference
for ii = 1:Nx
	% diagonal element
	I(count) = rowCount; II(count) = ii;
	V(count) = w2(1)*coef_dxx ;
	count = count + 1;
	% off-diagonal elements
	for q = 1:FDn
		% ii + q
		I(count) = rowCount; II(count) = ii+q;
		V(count) = w2(q+1)*coef_dxx;
		count = count + 1;
		% ii - q
		I(count) = rowCount; II(count) = ii-q;
		V(count) = w2(q+1)*coef_dxx;
		count = count + 1;

	end
	rowCount = rowCount + 1;
end

if S.BCx == 1
	% Removing outside domain entries (for periodic code this is unnecessary)
	isIn = (II >= 1) & (II <= Nx);
	S.I_11 = I(isIn); S.II_11 = II(isIn); S.V_11 = V(isIn);
elseif S.BCx == 0
	S.isOutl_11 = (II<1); S.isOutr_11 = (II>Nx); % Warning: Assumed influence of only neighboring cells
	S.I_11 = I; S.II_11 = mod(II+(Nx-1),Nx)+1; S.V_11 = V;
end

% D_yy laplacian in 1D
%-----------------------

% Initial number of non-zeros: including ghost nodes
nnzCount = (2 * FDn + 1) * Ny;

% Row and column indices and the corresponding non-zero values
% used to generate sparse matrix DL22 s.t. DL22(I(k),II(k)) = V(k)
I = zeros(nnzCount,1);
V = zeros(nnzCount,1);
II = zeros(nnzCount,1);
rowCount = 1;
count = 1;
coef_dyy = 1/dy^2;

% Find non-zero entries that use forward difference
for ii = 1:Ny
	% diagonal element
	I(count) = rowCount; II(count) = ii;
	V(count) = w2(1)*coef_dyy;
	count = count + 1;
	% off-diagonal elements
	for q = 1:FDn
		% ii + q
		I(count) = rowCount; II(count) = ii+q;
		V(count) = w2(q+1)*coef_dyy;
		count = count + 1;
		% ii - q
		I(count) = rowCount; II(count) = ii-q;
		V(count) = w2(q+1)*coef_dyy;
		count = count + 1;

	end
	rowCount = rowCount + 1;
end

if S.BCy == 1
	% Removing outside domain entries (for periodic code this is unnecessary)
	isIn = (II >= 1) & (II <= Ny);
	S.I_22 = I(isIn); S.II_22 = II(isIn); S.V_22 = V(isIn);
elseif S.BCy == 0
	S.isOutl_22 = (II<1); S.isOutr_22 = (II>Ny); % Warning: Assumed influence of only neighboring cells
	S.I_22 = I;  S.II_22 = mod(II+(Ny-1),Ny)+1; S.V_22 = V;
end

% D_zz laplacian in 1D
%-----------------------

% Initial number of non-zeros: including ghost nodes
nnzCount = (2 * FDn + 1) * Nz;

% Row and column indices and the corresponding non-zero values
% used to generate sparse matrix DL33 s.t. DL33(I(k),II(k)) = V(k)
I = zeros(nnzCount,1);
V = zeros(nnzCount,1);
II = zeros(nnzCount,1);
rowCount = 1;
count = 1;
coef_dzz = 1/dz^2;

% Find non-zero entries that use forward difference
for ii = 1:Nz
	% diagonal element
	I(count) = rowCount; II(count) = ii;
	V(count) = w2(1)*coef_dzz ;
	count = count + 1;
	% off-diagonal elements
	for q = 1:FDn
		% ii + q
		I(count) = rowCount; II(count) = ii+q;
		V(count) = w2(q+1)*coef_dzz;
		count = count + 1;
		% ii - q
		I(count) = rowCount; II(count) = ii-q;
		V(count) = w2(q+1)*coef_dzz;
		count = count + 1;

	end
	rowCount = rowCount + 1;
end

if S.BCz == 1
	% Removing outside domain entries (for periodic code this is unnecessary)
	isIn = (II >= 1) & (II <= Nz);
	S.I_33 = I(isIn); S.II_33 = II(isIn); S.V_33 = V(isIn);
elseif S.BCz == 0
	S.isOutl_33 = (II<1); S.isOutr_33 = (II>Nz); % Warning: Assumed influence of only neighboring cells
	S.I_33 = I; S.II_33 = mod(II+(Nz-1),Nz)+1; S.V_33 = V;
end

if S.cell_typ == 2
	% Create 1D gradient in all directions
	%---------------------------------------

	% x-direction
	%-------------
	nnz_x = 2*FDn*Nx;
	G = zeros(nnz_x,1);
	R = zeros(nnz_x,1);
	A = zeros(nnz_x,1);
	rowCount = 1;
	count = 1;
	coef_dx = 1/dx;

	for ii = 1:Nx
		for q = 1:FDn
			% ii + q
			G(count) = rowCount; R(count) = ii+q;
			A(count) = w1(q+1)*coef_dx;
			count = count + 1;
			% ii - q
			G(count) = rowCount; R(count) = ii-q;
			A(count) = -w1(q+1)*coef_dx;
			count = count + 1;
		end
		rowCount = rowCount + 1;
	end

	if S.BCx == 1
		% Removing outside domain entries (for periodic code this is unnecessary)
		isIn = (R >= 1) & (R <= Nx);
		S.I_1 = G(isIn); S.II_1 = R(isIn); S.V_1 = A(isIn);
	elseif S.BCx == 0
		S.isOutl_1 = (R<1); S.isOutr_1 = (R>Nx); % Warning: Assumed influence of only neighboring cells
		S.I_1 = G; S.II_1 = mod(R+(Nx-1),Nx)+1; S.V_1 = A;
	end

	% y-direction
	%-------------

	nnz_y = 2*FDn*Ny;
	G = zeros(nnz_y,1);
	R = zeros(nnz_y,1);
	A = zeros(nnz_y,1);
	count =1;
	rowCount =1;
	coef_dy = 1/dy;

	for jj = 1:Ny
		for q = 1:FDn
			% jj + q
			G(count) = rowCount; R(count) = jj+q;
			A(count) = w1(q+1)*coef_dy;
			count = count + 1;
			% jj - q
			G(count) = rowCount; R(count) = jj-q;
			A(count) = -w1(q+1)*coef_dy;
			count = count + 1;
		end
		rowCount = rowCount + 1;
	end

	if S.BCy == 1
		% Removing outside domain entries (for periodic code this is unnecessary)
		isIn = (R >= 1) & (R <= Ny);
		S.I_2 = G(isIn); S.II_2 = R(isIn); S.V_2 = A(isIn);
	elseif S.BCy == 0
		S.isOutl_2 = (R<1); S.isOutr_2 = (R>Ny);
		S.I_2 = G; S.II_2 = mod(R+(Ny-1),Ny)+1; S.V_2 = A;
	end


	% z-direction
	%-------------

	nnz_z = 2*FDn*Nz;
	G = zeros(nnz_z,1);
	R = zeros(nnz_z,1);
	A = zeros(nnz_z,1);
	count =1;
	rowCount =1;
	coef_dz = 1/dz;

	for kk = 1:Nz
		for q = 1:FDn
			% kk + q
			G(count) = rowCount; R(count) = kk+q;
			A(count) = w1(q+1)*coef_dz;
			count = count + 1;
			% kk - q
			G(count) = rowCount; R(count) = kk-q;
			A(count) = -w1(q+1)*coef_dz;
			count = count + 1;
		end
		rowCount = rowCount + 1;
	end

	if S.BCz == 1
		% Removing outside domain entries (for periodic code this is unnecessary)
		isIn = (R >= 1) & (R <= Nz);
		S.I_3 = G(isIn); S.II_3 = R(isIn); S.V_3 = A(isIn);
	elseif S.BCz == 0
		S.isOutl_3 = (R<1); S.isOutr_3 = (R>Nz);
		S.I_3 = G; S.II_3 = mod(R+(Nz-1),Nz)+1; S.V_3 = A;
	end
end


end



