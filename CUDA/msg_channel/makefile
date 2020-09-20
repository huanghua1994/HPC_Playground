EXE = msg_channel.x

CC     = gcc
CFLAGS = -O3 -Wall -fopenmp -g -std=gnu99 -march=native

#CC     = icc
#CFLAGS = -O3 -Wall -fopenmp -g -std=gnu99 -xHost

CUDA_PATH ?= /usr/local/cuda-10.0
NVCC       = nvcc
NVCCFLAGS  = -O3 -g -Wno-deprecated-gpu-targets -arch=sm_60
NVLDFLAGS  = -lcuda -lcudart -lcusparse

LDFLAGS = -fopenmp -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcusparse

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.o)

CU_SRCS = $(wildcard *.cu)
CU_OBJS = $(CU_SRCS:.cu=.cuo)

OBJS    = $(C_OBJS) $(CU_OBJS)

EXE: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -dlink -o msg_channel.o *.cuo.o $(NVLDFLAGS)
	$(CC) $(C_OBJS) *.cuo.o msg_channel.o -o $(EXE) $(LDFLAGS) 

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $^

%.cuo: %.cu
	$(NVCC) $(NVCCFLAGS) -rdc=true -o $@.o -c $^ 

clean:
	rm $(OBJS) $(EXE)
