EXE     = test_MKL_FFT_unpack.x
OBJS    = unpackMKLDftiForward.o test.o

HOSTCC  = icc
CFLAGS  = -O3 -Wall -g -std=gnu99 -qopenmp -xHost -mkl

EXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(CFLAGS) 
    
unpackMKLDftiForward.o: unpackMKLDftiForward.h unpackMKLDftiForward.c
	$(HOSTCC) $(CFLAGS) unpackMKLDftiForward.c -c
    
test.o: unpackMKLDftiForward.h test.c
	$(HOSTCC) $(CFLAGS) test.c -c
    
clean:
	rm *.o $(EXE)
