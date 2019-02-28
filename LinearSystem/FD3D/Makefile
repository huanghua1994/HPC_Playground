EXE     = FD3D_LinSys.x
OBJS    = AAR.o BiCGStab.o CG.o FD3D.o main.o 

HOSTCC  = icc
CFLAGS  = -O3 -Wall -qopenmp -g -std=gnu99 -xHost -mkl

EXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
AAR.o: FD3D.h AAR.h AAR.c
	$(HOSTCC) $(CFLAGS) AAR.c -c 
    
BiCGStab.o: FD3D.h BiCGStab.h BiCGStab.c
	$(HOSTCC) $(CFLAGS) BiCGStab.c -c 

CG.o: FD3D.h CG.h CG.c
	$(HOSTCC) $(CFLAGS) CG.c -c 
    
FD3D.o: FD3D.h FD3D.c 
	$(HOSTCC) $(CFLAGS) FD3D.c -c 
    
main.o: AAR.h BiCGStab.h CG.h FD3D.h main.c 
	$(HOSTCC) $(CFLAGS) main.c -c 
    
clean:
	rm *.o $(EXE)
