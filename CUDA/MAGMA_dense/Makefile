EXE = MAGMA_solver_test

HOSTCC  = gcc
HOSTCXX = g++

CUDA_PATH       ?= /usr/local/cuda-12.0
CUDA_INC_FLAGS  = -I$(CUDA_PATH)/include
CUDA_LD_FLAGS   = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -lcusolver -lcusparse

MAGMA_PATH      = $(HOME)/scratch.huah_gpu/magma-git
MAGMA_INC_FLAGS = -I$(MAGMA_PATH)/include
MAGMA_LD_FLAGS  = -L$(MAGMA_PATH)/lib -lmagma

CFLAGS  = -O2 -march=native -fopenmp -g -Wall $(MAGMA_INC_FLAGS) 
LDFLAGS = -O2 -march=native -fopenmp -g -Wall $(MAGMA_LD_FLAGS)  

all: magma_dsyevd.exe

magma_dsyevd.cpp.o: magma_dsyevd.cpp
	$(HOSTCXX) $(CFLAGS) $^ -c -o $@ 

magma_dsyevd.exe: magma_dsyevd.cpp.o
	$(HOSTCXX) $^ -o $@ $(LDFLAGS) 

clean:
	rm *.o *.exe