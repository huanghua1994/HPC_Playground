EXE  = CSR_SpMV_test.exe
OBJS = mmio.o utils.o CSRPlus_kernels.o CSRPlus_utils.o CSR_SpMV_test.o

CC      = icc
CFLAGS  = -O3 -Wall -g -std=gnu99 -qopenmp -xHost -mkl
LDFLAGS = -O3 -g -qopenmp -xHost -mkl

EXE: $(OBJS)
	$(CC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
mmio.o: makefile mmio.h mmio.c
	$(CC) $(CFLAGS) mmio.c -c
    
utils.o: makefile utils.h utils.c
	$(CC) $(CFLAGS) utils.c -c

CSRPlus_kernels.o: makefile CSRPlus.h CSRPlus_kernels.c
	$(CC) $(CFLAGS) CSRPlus_kernels.c -c

CSRPlus_utils.o: makefile CSRPlus.h CSRPlus_utils.c
	$(CC) $(CFLAGS) CSRPlus_utils.c -c

CSR_SpMV_test.o: makefile mmio.h utils.h CSRPlus.h CSR_SpMV_test.c
	$(CC) $(CFLAGS) CSR_SpMV_test.c -c

clean:
	rm *.o $(EXE)
