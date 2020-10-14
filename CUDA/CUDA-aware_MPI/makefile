CC             = mpicc
CFLAGS         = -O3 -Wall -fopenmp -g -std=gnu99 
CUDA_PATH     ?= /usr/local/cuda-10.0
GENCODE_SM60   = -gencode arch=compute_60,code=sm_60
GENCODE_SM70   = -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS  = $(GENCODE_SM60) $(GENCODE_SM70)
NVCC           = nvcc
NVFLAGS        = -O3 -g -Wno-deprecated-gpu-targets -lcuda -lcudart -lcusparse $(GENCODE_FLAGS)
LDFLAGS        = -fopenmp -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcusparse

EXES    = basic.exe rma_update.exe latency.exe

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.o)

CU_SRCS = $(wildcard *.cu)
CU_OBJS = $(CU_SRCS:.cu=.cu.obj)

all: $(EXES)

basic.exe: $(CU_OBJS) basic.o
	$(CC) $^ -o $@ $(LDFLAGS)

rma_update.exe: $(CU_OBJS) rma_update.o
	$(CC) $^ -o $@ $(LDFLAGS)

latency.exe: $(CU_OBJS) latency.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $^

%.cu.obj: %.cu
	$(NVCC) $(NVFLAGS) -o $@ -c $^ 

clean:
	rm $(OBJS) $(EXES)
