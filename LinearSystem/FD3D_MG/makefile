EXE  = FD3D_Poisson_MG.exe
OBJS = CSR_mat.o gen_stencil_mat.o Poisson_multigrid.o CG.o FD3D_Poisson_MG.o

CC      = icc
CFLAGS  = -O3 -Wall -g -std=gnu99 -qopenmp -xHost -mkl
LDFLAGS = -O3 -g -qopenmp -xHost -mkl

EXE: $(OBJS)
	$(CC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
CSR_mat.o: makefile CSR_mat.h CSR_mat.h
	$(CC) $(CFLAGS) CSR_mat.c -c
    
gen_stencil_mat.o: makefile CSR_mat.h gen_stencil_mat.h gen_stencil_mat.c
	$(CC) $(CFLAGS) gen_stencil_mat.c -c

Poisson_multigrid.o: makefile CSR_mat.h gen_stencil_mat.h Poisson_multigrid.h Poisson_multigrid.c
	$(CC) $(CFLAGS) Poisson_multigrid.c -c

CG.o: makefile CG.h CG.c
	$(CC) $(CFLAGS) CG.c -c

FD3D_Poisson_MG.o: makefile Poisson_multigrid.h FD3D_Poisson_MG.c
	$(CC) $(CFLAGS) FD3D_Poisson_MG.c -c

clean:
	rm *.o $(EXE)
