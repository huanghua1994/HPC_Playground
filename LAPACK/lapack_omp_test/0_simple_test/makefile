# specify the path MKLROOT if necessary, e.g.
# MKLROOT = /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl

FFLAGS = 

# if need to run with DEBUG mode, add -DDEBUG
#CPPFLAGS = -m64 -I${MKLROOT}/include -I include/ -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -ldl -lrt -O3 -DDEBUG -fopenmp

CPPFLAGS = -m64 -I${MKLROOT}/include -I include/ -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lmkl_blacs_intelmpi_lp64 -lpthread -ldl -lrt -O3 -fopenmp -std=c99

#OMP_LAPACK_CFLAGS = -m64 -I${MKLROOT}/include -I include/ -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lmkl_blacs_intelmpi_lp64 -lpthread -ldl -lrt -O3 -fopenmp

#CPPFLAGS = -lrt -O3 -fopenmp

FPPFLAGS = 


#OBJSC = test_eigsolver_omp.o eigsolver_LAPACK.o
OBJSC = test_MPI_OMP.o

LIBBASE = test_mpi_omp

override CC=mpicc

all: test_mpi_omp

test_mpi_omp: $(OBJSC)
	$(CC) $(CPPFLAGS) -o $(LIBBASE) $^

.PHONY: clean
clean:
	rm -f  $(OBJSC) $(LIBBASE) 
