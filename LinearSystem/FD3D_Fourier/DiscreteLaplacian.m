function DL = DiscreteLaplacian(L1,L2,L3,Nx,Ny,Nz,n0,BC)
% @brief	DiscreteLaplacian(R,H,Nr,Ntheta,Nz,order,nu) discretizes the Laplacian
%			operator in a cuboidal domain (with homogeneous Dirichlet boundary condition
%			or periodic boundary condition) and returns the matrix DL.
%
% @param BC		Boundary condition: 1--isolated luster; 2--periodic system
% @param L1		The side of the domain in the x-direction
% @param L2		The side of the domain in the y-direction
% @param L3		The side of the domain in the z-direction
% @param Nx		The number of nodes in [0,L1]
% @param Ny		The number of nodes in [0,L2]
% @param Nz		The number of nodes in [0,L3]
% @param n0		Half of the order of the finite difference scheme
%
% @authors	Qimen Xu <qimenxu@gatech.edu>
%			Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
% @2016 Georgia Institute of Technology.
%

if (L1 <= 0)
	error('DiscreteLaplacian(S): ''S.L1'' must be positive')
elseif (L2 <= 0)
	error('DiscreteLaplacian(S): ''S.L2'' must be positive')
elseif (L3 <= 0)
	error('DiscreteLaplacian(S): ''S.L3'' must be positive')
elseif ((Nx - round(Nx) ~= 0)||(Nx <= 0))
	error('DiscreteLaplacian(S): ''S.Nx'' must be a positive integer')
elseif ((Ny - round(Ny) ~= 0)||(Ny <= 0))
	error('DiscreteLaplacian(S): ''S.Ny'' must be a positive integer')
elseif ((Nz - round(Nz) ~= 0)||(Nz <= 0))
	error('DiscreteLaplacian(S): ''S.Nz'' must be a positive integer')
elseif ((n0 - round(n0) ~= 0)||(n0 <= 0))
	error('DiscreteLaplacian(S): ''S.FDn'' must be a positive integer')	
end

% Total number of nodes
N = Nx * Ny * Nz;

% Mesh sizes
if BC == 1
	dx = L1 / (Nx - 1);
	dy = L2 / (Ny - 1);
	dz = L3 / (Nz - 1);
elseif BC == 2
	dx = L1 / Nx;
	dy = L2 / Ny;
	dz = L3 / Nz;
end

% Finite difference weights of the second derivative
w2 = zeros(1,n0+1) ; 
for k=1:n0
    w2(k+1) = (2*(-1)^(k+1))*(factorial(n0)^2)/...
                    (k*k*factorial(n0-k)*factorial(n0+k));
    w2(1) = w2(1)-2*(1/(k*k));
end

% Phase factor
% PhaseFactor = exp(-pi*1i*nu); % it is equivalent to (-1)^nu

% Initial number of non-zeros: including ghost nodes
nnzCount = (3 * (2 * n0) + 1) * N;

% Row and column indices and the corresponding non-zero values
% used to generate sparse matrix DL s.t. DL(I(k),J(k)) = V(k)
I = zeros(nnzCount,1);
J = zeros(nnzCount,1);
V = zeros(nnzCount,1);

% Indices of the columns J = II*Ntheta*(Nz+1) + JJ*(Nz+1) + KK + 1;
II = zeros(nnzCount,1);
JJ = zeros(nnzCount,1);
KK = zeros(nnzCount,1);

rowCount = 1;
count = 1;
dx2 = dx^2;
dy2 = dy^2;
dz2 = dz^2;

% Find non-zero entries that use forward difference 
for kk = 1:Nz
	for jj = 1:Ny
		for ii = 1:Nx
			% diagonal element
			I(count) = rowCount; II(count) = ii; JJ(count) = jj; KK(count) = kk;
			V(count) = w2(1)/(dx2) + w2(1)/(dy2) + w2(1)/(dz2);
			count = count + 1;
			% off-diagonal elements
			for q = 1:n0
                % ii + q
                I(count) = rowCount; II(count) = ii+q; JJ(count) = jj; KK(count) = kk;
                V(count) = w2(q+1)/(dx2);
                count = count + 1;
				% ii - q
				I(count) = rowCount; II(count) = ii-q; JJ(count) = jj; KK(count) = kk;
				V(count) = w2(q+1)/(dx2);
				count = count + 1;	
                % jj + q
                I(count) = rowCount; II(count) = ii; JJ(count) = jj+q; KK(count) = kk;
                V(count) = w2(q+1)/(dy2);
                count = count + 1;
                % jj - q
                I(count) = rowCount; II(count) = ii; JJ(count) = jj-q; KK(count) = kk;
                V(count) = w2(q+1)/(dy2);
                count = count + 1;
                % kk + q
                I(count) = rowCount; II(count) = ii; JJ(count) = jj; KK(count) = kk+q;
                V(count) = w2(q+1)/(dz2);
                count = count + 1;
                % kk - q
                I(count) = rowCount; II(count) = ii; JJ(count) = jj; KK(count) = kk-q;
                V(count) = w2(q+1)/(dz2);
                count = count + 1;
			end
			rowCount = rowCount + 1;
		end
	end
end


if(BC == 1)
	% Removing outside domain entries from r and z directions (for periodic code this is unnecessary)
	isIn = (II >= 1) & (II <= Nx) & (JJ >= 1) & (JJ <= Ny) & (KK >= 1) & (KK <= Nz);
	I = I(isIn); II = II(isIn); JJ = JJ(isIn); KK = KK(isIn); V = V(isIn);
elseif (BC == 2)
	II = mod(II-1+Nx,Nx)+1; JJ = mod(JJ-1+Ny,Ny)+1; KK = mod(KK-1+Nz,Nz)+1;
end

% Getting linear indices of the columns
J = (KK-1)*Nx*Ny + (JJ-1)*Nx + II;

% Create discretized Laplacian
DL = sparse(I,J,V,N,N);