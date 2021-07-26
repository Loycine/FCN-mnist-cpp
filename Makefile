NVCC = $(shell which nvcc)
CC = g++
OPT = -O2 -g
#OPT = -pg -O2
#OPT = -pg -fprofile-report -O2
NVCC_FLAGS = $(OPT) -G -Xcompiler -Wall
CC_FLAGS = $(OPT) -Wall -fopenmp -std=c++17 #-mavx512f -mfma -Wall

cblas_FCN = xcblas_FCN

all: $(cblas_FCN) #$(eigen_FCN) #

$(cblas_FCN): drv_fully_connected_nn.o fully_connected_layer.o fully_connected_nn.o #cblas_FCN.o #mt19937ar.o #im2col.o 
	@echo "----- Building $(eigen_FCN) -----"
	$(CC) $(CC_FLAGS) $^ -lopenblas -o $@
	@echo

%.o: %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@ 
%.o: %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@
%.o : %.cu 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o x* *.x
	
#$(CC) $(CC_FLAGS) $^ -lopenblas64 -o $@
#$(CC) $(CC_FLAGS) $^ -L/opt/intel/mkl/lib/intel64 -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_core -L/opt/intel/lib/intel64/ -liomp5 -o $@
#$(CC) $(CC_FLAGS) $^ -lopenblas64 -o $@
#$(CC) $(CC_FLAGS) $^ /opt/OpenBLAS/lib/libopenblas.a -o $@
	#$(CC) $(CC_FLAGS) $^ -lblas64 -lpthread -o $@
	#$(CC) $(CC_FLAGS) $^ /usr/lib/x86_64-linux-gnu/openblas64-openmp/libopenblas64.a -o $@
