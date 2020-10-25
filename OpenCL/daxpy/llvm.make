include ../common/llvm.defs.make

EXE     = ocl_daxpy
CPPSRCS = $(wildcard *.cpp)
OBJS    = $(CPPSRCS:.cpp=.cpp.o) ../common/ocl_utils.o

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $(EXE) 

../common/ocl_utils.o: ../common/ocl_utils.c
	$(CC) $(CFLAGS) ../common/ocl_utils.c -c -o ../common/ocl_utils.o

%.cpp.o: %.cpp
	$(CXX) $(CFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXE)