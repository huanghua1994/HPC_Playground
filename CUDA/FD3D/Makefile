FD3DEXE = FD3D.x

#HOSTCC  = gcc
#CFLAGS  = -O3 -Wall -fopenmp -g -std=gnu99 -march=native

HOSTCC  = icc
CFLAGS  = -O3 -Wall -qopenmp -g -std=gnu99 -xHost

CUPATH  = /usr/local/cuda-8.0
NVCC    = nvcc
NVFLAGS = -O3 -g -Wno-deprecated-gpu-targets -lcudart_static -lcusparse_static -lculibos -arch=sm_60

LDFLAGS = -fopenmp -I $(CUPATH)/include -L$(CUPATH)/lib64 -lcudart -lcusparse

OBJS    = FD3D.o stencil_3axis_ref.o stencil_3axis_cuda.o 

FD3DEXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(FD3DEXE) $(LDFLAGS) 
    
stencil_3axis_ref.o: stencil_3axis_ref.c
	$(HOSTCC) $(CFLAGS) stencil_3axis_ref.c -c 

FD3D.o: FD3D.c stencil_3axis_ref.h stencil_3axis_cuda.h
	$(HOSTCC) $(CFLAGS) FD3D.c -c 
    
stencil_3axis_cuda.o: stencil_3axis_cuda_kernel.cuh stencil_3axis_cuda.cu CUDA_Utils.h
	$(NVCC) $(NVFLAGS) stencil_3axis_cuda.cu -c 
    
clean:
	rm *.o $(FD3DEXE)
