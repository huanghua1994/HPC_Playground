EXE    = FD3D_Poisson.exe
OBJS   = AAR.o BiCGStab.o CG.o FD3D.o main.o 

CC     = icc
CFLAGS = -O3 -Wall -qopenmp -g -std=gnu99 -xHost -mkl

EXE: $(OBJS)
	$(CC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
AAR.o: FD3D.h AAR.h AAR.c
	$(CC) $(CFLAGS) AAR.c -c 
    
BiCGStab.o: FD3D.h BiCGStab.h BiCGStab.c
	$(CC) $(CFLAGS) BiCGStab.c -c 

CG.o: FD3D.h CG.h CG.c
	$(CC) $(CFLAGS) CG.c -c 
    
FD3D.o: FD3D.h FD3D.c 
	$(CC) $(CFLAGS) FD3D.c -c 
    
main.o: AAR.h BiCGStab.h CG.h FD3D.h main.c 
	$(CC) $(CFLAGS) main.c -c 
    
clean:
	rm *.o $(EXE)
