EXES = test_cublas_os.exe

CU_SRCS = $(wildcard *.cu)
CU_OBJS = $(CU_SRCS:.cu=.cu.o)
OBJS    = $(CU_OBJS)
LIBS    = -lcuda -lcudart -lcublas -lcublasLt -lcusolver -lcurand -lm

GENCODE_SM70  = -gencode arch=compute_70,code=sm_70
GENCODE_SM75  = -gencode arch=compute_75,code=sm_75
GENCODE_SM80  = -gencode arch=compute_80,code=sm_80
GENCODE_SM86  = -gencode arch=compute_86,code=sm_86
GENCODE_SM89  = -gencode arch=compute_89,code=sm_89
GENCODE_SM90  = -gencode arch=compute_90,code=sm_90
GENCODE_SM90A = -gencode arch=compute_90a,code=sm_90a
GENCODE_FLAGS  = $(GENCODE_SM70) $(GENCODE_SM75)
GENCODE_FLAGS += $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM89)
GENCODE_FLAGS += $(GENCODE_SM90) $(GENCODE_SM90A)

CUDA_PATH   ?= /usr/local/cuda-12.0
NVCC        = nvcc
NVCCFLAGS   = -O3 -g -Xcompiler=-fopenmp $(GENCODE_FLAGS)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(OBJS)

all: $(EXES)

%.exe: %.cu.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

test_cublas_os.cu.o: test_cublas_os.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c test_cublas_os.cu

clean:
	rm *.cu.o *.exe
