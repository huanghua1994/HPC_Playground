EXE     = MKL_FFT_fwd_unpack.x
OBJS    = test.o

HOSTCC  = icc
CFLAGS  = -O3 -Wall -g -std=gnu99 -qopenmp -xHost -mkl

EXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
test.o: test.c
	$(HOSTCC) $(CFLAGS) test.c -c
    
clean:
	rm *.o $(EXE)
