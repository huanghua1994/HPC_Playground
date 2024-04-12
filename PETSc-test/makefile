MPICC ?= mpicc

PETSC_INSTALL_DIR = /storage/scratch1/8/hhuang368/petsc-3.20.5/install
CFLAGS  = -I $(PETSC_INSTALL_DIR)/include -Wall -g -std=gnu11 -O2 -fopenmp
LDFLAGS = -L $(PETSC_INSTALL_DIR)/lib -g -O2 -fopenmp -lm

.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: test_petsc_spmv.exe

%.c.o: %.c
	$(MPICC) $(CFLAGS) -c $^ -o $@

test_petsc_spmv.exe: mmio.c.o mmio_utils.c.o test_petsc_spmv.c.o
	$(MPICC) -o $@ $^ $(LDFLAGS) -lpetsc

clean:
	rm -f *.exe *.c.o
