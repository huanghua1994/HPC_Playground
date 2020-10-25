EXE       = ocl_saxpy
NVCC      = nvcc
NVCCFLAGS = -O3 -g -Wno-deprecated-gpu-targets
LDFLAGS   = -O3 -lOpenCL 

CPPSRCS = $(wildcard *.cpp)
OBJS    = $(CPPSRCS:.cpp=.cpp.o) ../common/ocl_utils.o

all: $(EXE)

$(EXE): $(OBJS)
	$(NVCC) $(LDFLAGS) $(OBJS) -o $(EXE) 

../common/ocl_utils.o: ../common/ocl_utils.c
	$(NVCC) $(NVCCFLAGS) ../common/ocl_utils.c -c -o ../common/ocl_utils.o

%.cpp.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXE)