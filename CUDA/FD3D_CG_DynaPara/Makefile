EXE     = CUDA_FD3D_CG.x
OBJS    = CG.o FD3D.o FD3D_CG_cuda_link.o FD3D_CG_cuda.o  main.o 

HOSTCC  = icpc
CFLAGS  = -O3 -Wall -qopenmp -g -std=gnu99 -xHost

CUPATH  = /usr/local/cuda-10.0
NVCC    = nvcc
NVFLAGS = -O3 -g -Wno-deprecated-gpu-targets -lcudart -lculibos -arch=sm_61 -rdc=true

LDFLAGS = -I $(CUPATH)/include -L$(CUPATH)/lib64 -lcudart -lculibos -lcudadevrt -qopenmp 

EXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(LDFLAGS) 

CG.o: FD3D.h CG.h CG.c
	$(HOSTCC) $(CFLAGS) CG.c -c 
    
FD3D.o: FD3D.h FD3D.c 
	$(HOSTCC) $(CFLAGS) FD3D.c -c 

FD3D_CG_cuda.o: FD3D_CG_cuda.h FD3D_CG_cuda.cu FD3D_CG_cuda_kernels.cuh
	$(NVCC) $(NVFLAGS) FD3D_CG_cuda.cu -c 
    
FD3D_CG_cuda_link.o: FD3D_CG_cuda.o 
	$(NVCC) -dlink -o FD3D_CG_cuda_link.o FD3D_CG_cuda.o -lcudadevrt -lcudart -arch=sm_61 
    
main.o: CG.h FD3D.h main.c 
	$(HOSTCC) $(CFLAGS) main.c -c 
    
clean:
	rm *.o $(EXE)
