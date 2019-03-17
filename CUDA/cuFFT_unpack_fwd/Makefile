EXE     = cuFFT_fwd_unpack.x

HOSTCC  = icc
CFLAGS  = -O3 -Wall -qopenmp -g -std=gnu99 -xHost

NVCC    = nvcc
CUPATH  = /usr/local/cuda-10.0
NVFLAGS = -O3 -g -Wno-deprecated-gpu-targets -lcudart_static -lcufft_static -lculibos -arch=sm_60

LDFLAGS = -qopenmp -I $(CUPATH)/include -L$(CUPATH)/lib64 -lcudart -lcufft -lculibos 

OBJS    = test.o 

FD3DEXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(LDFLAGS) 
    
test.o: test.cu
	$(NVCC) $(NVFLAGS) test.cu -c 
    
clean:
	rm *.o $(FD3DEXE)
