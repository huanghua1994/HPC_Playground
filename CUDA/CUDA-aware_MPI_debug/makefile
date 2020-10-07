ifdef MPI_ROOT
MPI_HOME ?= $(MPI_ROOT)
endif

ifdef CUDA_ROOT
CUDA_HOME ?= $(CUDA_ROOT)
endif

ifndef MPI_HOME
$(error MPI_HOME is not set)
endif

ifndef CUDA_HOME
$(error CUDA_HOME is not set)
endif

GENCODE_SM60  = -gencode arch=compute_60,code=sm_60
GENCODE_SM70  = -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS = $(GENCODE_SM60) $(GENCODE_SM70)

NVCC          = nvcc
NVCC_FLAGS    = -dc -Xcompiler -fopenmp -lineinfo $(GENCODE_FLAGS) -std=c++11 -I$(MPI_HOME)/include -g
NVCC_LDFLAGS  = -ccbin=mpicxx --compiler-options -fopenmp -L$(MPI_HOME)/lib -lmpi -L$(CUDA_HOME)/lib64 -lcuda -lcudart

EXES = sendrecv.exe send_recv.exe put_and_acc.exe

all: $(EXES)

sendrecv.exe: makefile sendrecv.cu
	$(NVCC) $(NVCC_FLAGS) sendrecv.cu -c -o sendrecv.o
	$(NVCC) $(GENCODE_FLAGS) sendrecv.o -o sendrecv.exe $(NVCC_LDFLAGS)

send_recv.exe: makefile send_recv.cu
	$(NVCC) $(NVCC_FLAGS) send_recv.cu -c -o send_recv.o
	$(NVCC) $(GENCODE_FLAGS) send_recv.o -o send_recv.exe $(NVCC_LDFLAGS)

put_and_acc.exe: makefile put_and_acc.cu
	$(NVCC) $(NVCC_FLAGS) put_and_acc.cu -c -o put_and_acc.o
	$(NVCC) $(GENCODE_FLAGS) put_and_acc.o -o put_and_acc.exe $(NVCC_LDFLAGS)

.PHONY.: clean
clean:
	rm -f *.exe *.o
