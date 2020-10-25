include ../common/nvcc.defs.make

EXE     = ocl_daxpy
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