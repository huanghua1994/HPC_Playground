SPMVEXE = cuda_CSR_SpMV.x

#HOSTCC  = gcc
#CFLAGS  = -O3 -march=native -Wall -fopenmp -g -std=gnu99

HOSTCC   = icc
CFLAGS   = -O3 -xHost -Wall -qopenmp -g -std=gnu99

CUPATH  = /usr/local/cuda-8.0
NVCC    = nvcc
NVFLAGS = -O3 -g -Wno-deprecated-gpu-targets -lcudart_static -lcusparse_static -lculibos -arch=sm_60 

LDFLAGS = -fopenmp -I $(CUPATH)/include -L$(CUPATH)/lib64 -lcudart -lcusparse

OBJS    = CSRPlus_utils.o CSRPlus_kernels.o CSR_SpMV.o cuSPARSE_SpMV_test.o 

SPMVEXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(SPMVEXE) $(LDFLAGS) 

CSR_SpMV.o: CSR_SpMV.c CSRPlus.h
	$(HOSTCC) $(CFLAGS) CSR_SpMV.c -c 

CSRPlus_utils.o: CSRPlus.h CSRPlus_utils.c
	$(HOSTCC) $(CFLAGS) CSRPlus_utils.c -c 

CSRPlus_kernels.o: CSRPlus.h CSRPlus_kernels.c
	$(HOSTCC) $(CFLAGS) CSRPlus_kernels.c -c 

cuSPARSE_SpMV_test.o: cuSPARSE_SpMV_test.cu cuSPARSE_SpMV_test.h CUDA_Utils.h
	$(NVCC) $(NVFLAGS) cuSPARSE_SpMV_test.cu -c 

clean:
	rm *.o $(SPMVEXE)
