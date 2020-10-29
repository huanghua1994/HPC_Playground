EXES    = mpi_dev_mem.exe
MPICC   = mpicc
MPICXX  = mpicxx
CSRCS   = $(wildcard *.c)
COBJS   = $(CSRCS:.c=.c.o)
CPPSRCS = $(wildcard *.cpp)
CPPOBJS = $(CPPSRCS:.cpp=.cpp.o)
OBJS    = $(COBJS) $(CPPOBJS)

LIBS += -lsycl

.SECONDARY: $(OBJS)

all: $(EXES)

%.exe: $(OBJS)
	$(MPICXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.c.o: %.c
	$(MPICC) $(CFLAGS) -c $^ -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXES)