include ../common/llvm.defs.make
CXXFLAGS += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
LDFLAGS  += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice

EXES    = mpi-cuda-ipc.exe
NVCC    = nvcc
MPICC   = mpicc
CSRCS   = $(wildcard *.c)
COBJS   = $(CSRCS:.c=.c.o)
CUSRCS  = $(wildcard *.cu)
CUOBJS  = $(CUSRCS:.cu=.cu.o)
CPPSRCS = $(wildcard *.cpp)
CPPOBJS = $(CPPSRCS:.cpp=.cpp.o)
OBJS    = $(COBJS) $(CUOBJS) $(CPPOBJS)

CUDA_PATH ?= /usr/common/software/cuda/10.2.89
CUDA_LIB   = -lcuda -lcudart
NVCCFLAGS  = -O3 -g --compiler-options -fPIC

MPI_PATH  ?= /usr/common/software/mvapich2/2.3.2/gcc/8.3.0/cuda/10.2.89
MPI_LIB    = -lmpi

LIBS += -lsycl -L$(MPI_PATH)/lib $(MPI_LIB) -L$(CUDA_PATH)/lib64 $(CUDA_LIB)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(OBJS)

all: $(EXES)

%.exe: $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.c.o: %.c
	$(MPICC) $(CFLAGS) -c $^ -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXES)