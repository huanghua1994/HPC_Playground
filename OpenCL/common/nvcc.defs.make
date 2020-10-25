CC      = nvcc
CXX     = nvcc
CFLAGS  = -O3 -g -Wno-deprecated-gpu-targets
LDFLAGS = -O3
LIBS    = -lOpenCL

GENCODE_SM60    = -gencode arch=compute_60,code=sm_60
GENCODE_SM70    = -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS   = $(GENCODE_SM60) $(GENCODE_SM70)
CFLAGS         += $(GENCODE_FLAGS)