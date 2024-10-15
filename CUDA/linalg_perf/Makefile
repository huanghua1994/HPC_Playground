EXES = gemm_test.exe lu_chol_test.exe qr_test.exe mgs2_test.exe \
       syevd_test.exe bgj_test.exe

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
CU_SRCS = $(wildcard *.cu)
CU_OBJS = $(CU_SRCS:.cu=.cu.o)
OBJS    = $(C_OBJS) $(CU_OBJS)

CC      = gcc
DEFS    = 
INCS    = 
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)
LDFLAGS = -L$(CUDA_PATH)/lib64
LIBS    = -lcuda -lcudart -lcublas -lcusolver -lcurand -lm

GENCODE_SM60  = -gencode arch=compute_60,code=sm_60
GENCODE_SM70  = -gencode arch=compute_70,code=sm_70
GENCODE_SM80  = -gencode arch=compute_80,code=sm_80
GENCODE_SM90  = -gencode arch=compute_90,code=sm_90
GENCODE_FLAGS = $(GENCODE_SM70) $(GENCODE_SM80) $(GENCODE_SM90)

CUDA_PATH   ?= /usr/local/cuda-10.0
NVCC        = nvcc
NVCCFLAGS   = -O3 -g --compiler-options -fPIC $(GENCODE_FLAGS)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(OBJS)

all: $(EXES)

%.exe: %.cu.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $^

clean:
	rm $(OBJS) $(EXES)
