EXE  = CSR_SpMV_test.exe
OBJS = mmio.o utils.o CSRPlus.o CSR_SpMV_test.o

CC      = icc
CFLAGS  = -O3 -Wall -g -std=gnu99 -qopenmp -xHost -mkl
LDFLAGS = -O3 -g -qopenmp -xHost -mkl

EXE: $(OBJS)
	$(CC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
mmio.o: makefile mmio.h mmio.c
	$(CC) $(CFLAGS) mmio.c -c
    
utils.o: makefile utils.h utils.c
	$(CC) $(CFLAGS) utils.c -c

CSRPlus.o: makefile CSRPlus.h CSRPlus.c
	$(CC) $(CFLAGS) CSRPlus.c -c

CSR_SpMV_test.o: makefile mmio.h utils.h CSRPlus.h CSR_SpMV_test.c
	$(CC) $(CFLAGS) CSR_SpMV_test.c -c

clean:
	rm *.o $(EXE)
