MPICC ?= mpicc

PETSC_INSTALL_DIR = /storage/scratch1/8/hhuang368/petsc-3.20.5/install
CFLAGS  = -I $(PETSC_INSTALL_DIR)/include -Wall -g -std=gnu11 -O2 -fopenmp
LDFLAGS = -L $(PETSC_INSTALL_DIR)/lib -lpetsc -g -O2 -fopenmp -lm

SHARED_OBJS = mmio.c.o mmio_utils.c.o

.SUFFIXES:

.SECONDARY: $(SHARED_OBJS)

all: test_petsc_spmv.exe test_petsc_spmm.exe

%.c.o: %.c
	$(MPICC) $(CFLAGS) -c $^ -o $@

%.exe: $(SHARED_OBJS) %.c.o
	$(MPICC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.exe *.c.o
