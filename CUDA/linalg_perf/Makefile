EXES = bgj_test.exe lu_chol_test.exe syevd_test.exe \
       gemm_test.exe cublaslt_gemm_test.exe mgs2_test.exe qr_test.exe

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

bgj_test.cu.o: bgj_test.cu cuda_utils.cuh cublas_cusolver_init.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ -c bgj_test.cu

lu_chol_test.cu.o: lu_chol_test.cu cuda_utils.cuh cublas_cusolver_init.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ -c lu_chol_test.cu

syevd_test.cu.o: syevd_test.cu cuda_utils.cuh cublas_cusolver_init.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ -c syevd_test.cu

gemm_test.cu.o: gemm_test.cu cuda_utils.cuh helper.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ -c gemm_test.cu

cublaslt_gemm_test.cu.o: cublaslt_gemm_test.cu cuda_utils.cuh 
	$(NVCC) $(NVCCFLAGS) -o $@ -c cublaslt_gemm_test.cu

mgs2_test.cu.o: mgs2_test.cu cuda_utils.cuh helper.cuh block_mgs.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ -c mgs2_test.cu

qr_test.cu.o: qr_test.cu cuda_utils.cuh helper.cuh householder_qr.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ -c qr_test.cu

clean:
	rm *.cu.o *.exe