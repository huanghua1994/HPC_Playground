EXE     = FD3D_Poisson.x
OBJS    = Poisson_FD_FFT.o Poisson_FFT_Solver.o

HOSTCC  = icc
CFLAGS  = -O3 -Wall -g -std=gnu99 -qopenmp -xHost -mkl

EXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
Poisson_FD_FFT.o: Poisson_FFT_Solver.h Poisson_FD_FFT.c
	$(HOSTCC) $(CFLAGS) Poisson_FD_FFT.c -c
    
Poisson_FFT_Solver.o: Poisson_FFT_Solver.h Poisson_FFT_Solver.c
	$(HOSTCC) $(CFLAGS) Poisson_FFT_Solver.c -c
    
clean:
	rm *.o $(EXE)
