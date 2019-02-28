EXE     = FD3D_LinSys.x
OBJS    = FD3D.o BiCGStab.o CG.o main.o 

HOSTCC  = icc
CFLAGS  = -O3 -Wall -qopenmp -g -std=gnu99 -xHost

EXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
main.o: FD3D.h BiCGStab.h CG.h main.c 
	$(HOSTCC) $(CFLAGS) main.c -c 

FD3D.o: FD3D.h FD3D.c 
	$(HOSTCC) $(CFLAGS) FD3D.c -c 
    
BiCGStab.o: FD3D.h BiCGStab.h BiCGStab.c
	$(HOSTCC) $(CFLAGS) BiCGStab.c -c 

CG.o: FD3D.h CG.h CG.c
	$(HOSTCC) $(CFLAGS) CG.c -c 
    
clean:
	rm *.o $(EXE)
