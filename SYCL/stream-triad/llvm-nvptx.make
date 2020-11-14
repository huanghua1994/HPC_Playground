include ../common/llvm.defs.make
CXXFLAGS += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
LDFLAGS  += -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
include stream-triad.make