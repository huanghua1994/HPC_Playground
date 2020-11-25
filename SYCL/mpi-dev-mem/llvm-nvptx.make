include ../common/llvm.defs.make
CXXFLAGS += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
LDFLAGS  += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
include mpi_dev_mem.make