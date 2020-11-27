EXES    = mpi_dev_mem.exe
MPICC   = mpicc
CSRCS   = $(wildcard *.c)
COBJS   = $(CSRCS:.c=.c.o)
CPPSRCS = $(wildcard *.cpp)
CPPOBJS = $(CPPSRCS:.cpp=.cpp.o)
OBJS    = $(COBJS) $(CPPOBJS)

MPI_DIR ?= /usr/common/software/mvapich2/2.3.2/gcc/8.3.0/cuda/10.2.89
MPI_LIB  = -lmpi

LIBS += -lsycl -L$(MPI_DIR)/lib $(MPI_LIB)

.SECONDARY: $(OBJS)

all: $(EXES)

%.exe: $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.c.o: %.c
	$(MPICC) $(CFLAGS) -c $^ -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXES)