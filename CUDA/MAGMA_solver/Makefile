EXE = MAGMA_solver_test

HOSTCC = icc

CUDA_PATH       = /usr/local/cuda-10.0
CUDA_INC_FLAGS  = -I$(CUDA_PATH)/include
CUDA_LD_FLAGS   = -L$(CUDA_PATH)/lib64 -lcublas -lcudart -lcusparse

MAGMA_PATH      = $(HOME)/Workspace/magma-2.5.0
MAGMA_INC_FLAGS = -I$(MAGMA_PATH)/include -I$(MAGMA_PATH)/sparse/include 
MAGMA_LD_FLAGS  = -L$(MAGMA_PATH)/lib -lmagma -lmagma_sparse 

CFLAGS  = -O3 -xHost -qopenmp -g -Wall $(MAGMA_INC_FLAGS) $(CUDA_INC_FLAGS)
LDFLAGS = -O3 -xHost -qopenmp -g -Wall $(MAGMA_LD_FLAGS)  $(CUDA_LD_FLAGS)

OBJS = mmio.c magma_csr_cg.o CSRPlus_utils.o CSRPlus_kernels.o

EXE: $(OBJS)
	$(HOSTCC) $(OBJS) -o $(EXE) $(LDFLAGS) 

CSRPlus_utils.o: CSRPlus.h CSRPlus_utils.c
	$(HOSTCC) $(CFLAGS) CSRPlus_utils.c -c 

CSRPlus_kernels.o: CSRPlus.h CSRPlus_kernels.c
	$(HOSTCC) $(CFLAGS) CSRPlus_kernels.c -c 
    
mmio.o: mmio.c
	$(HOSTCC) $(CFLAGS) mmio.c -c
    
magma_csr_cg.o: magma_csr_cg.c
	$(HOSTCC) $(CFLAGS) magma_csr_cg.c -c

clean:
	rm *.o $(EXE)