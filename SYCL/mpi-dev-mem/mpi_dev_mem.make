EXES    = mpi_dev_mem.exe
MPICC   = mpicc
CSRCS   = $(wildcard *.c)
COBJS   = $(CSRCS:.c=.c.o)
CPPSRCS = $(wildcard *.cpp)
CPPOBJS = $(CPPSRCS:.cpp=.cpp.o)
OBJS    = $(COBJS) $(CPPOBJS)

MPI_LIB_DIR  = /opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/lib/release
MPI_LIB_DIR += /opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/lib
MPI_LIB      = -lmpi

LIBS += -lsycl -L$(MPI_LIB_DIR) $(MPI_LIB)

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