SPMVEXE = cuda_CSR_SpMV.x

HOSTCC  = gcc
CFLAGS  = -O3 -march=native -Wall -fopenmp -g -std=gnu99

#HOSTCC   = icc
#CFLAGS   = -O3 -xHost -Wall -qopenmp -g -std=gnu99

CUPATH  = /usr/local/cuda-8.0
NVCC    = nvcc
NVFLAGS = -O3 -g -Wno-deprecated-gpu-targets -lcudart_static -lcusparse_static -lculibos -arch=sm_60 

LDFLAGS = -fopenmp -I $(CUPATH)/include -L$(CUPATH)/lib64 -lcudart -lcusparse

OBJS    = CSRPlus.o CSR_SpMV.o cuSPARSE_SpMV_test.o 

SPMVEXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(SPMVEXE) $(LDFLAGS) 
    
CSR_SpMV.o: CSR_SpMV.c CSRPlus.h
	$(HOSTCC) $(CFLAGS) CSR_SpMV.c -c 

CSRPlus.o: CSRPlus.c CSRPlus.h
	$(HOSTCC) $(CFLAGS) CSRPlus.c -c 
    
cuSPARSE_SpMV_test.o: cuSPARSE_SpMV_test.cu cuSPARSE_SpMV_test.h CUDA_Utils.h
	$(NVCC) $(NVFLAGS) cuSPARSE_SpMV_test.cu -c 
    
clean:
	rm *.o $(SPMVEXE)
