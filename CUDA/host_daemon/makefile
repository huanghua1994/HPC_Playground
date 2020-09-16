EXE = host_daemon.x

CC     = gcc
CFLAGS = -O3 -Wall -fopenmp -g -std=gnu99 -march=native

#CC     = icc
#CFLAGS = -O3 -Wall -fopenmp -g -std=gnu99 -xHost

CUDA_PATH ?= /usr/local/cuda-10.0
NVCC       = nvcc
NVFLAGS    = -O3 -g -Wno-deprecated-gpu-targets -lcudart_static -lcusparse_static -lculibos -arch=sm_60

LDFLAGS = -fopenmp -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcusparse

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.o)

CU_SRCS = $(wildcard *.cu)
CU_OBJS = $(CU_SRCS:.cu=.cuo)

OBJS    = $(C_OBJS) $(CU_OBJS)

EXE: $(OBJS)
	$(CC) $(OBJS) -o $(EXE) $(LDFLAGS) 

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $^

%.cuo: %.cu
	$(NVCC) $(NVFLAGS) -o $@ -c $^ 

clean:
	rm $(OBJS) $(EXE)
