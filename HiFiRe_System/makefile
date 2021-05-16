CC = /opt/nvidia/hpc_sdk/Linux_x86_64/21.2/compilers/bin/pgc++
CFLAGS = -c
LIBB = -L/usr/local/lib -lgeos
LIBRA = lib/libshp.a lib/librtree.a
DEBUG =

all: cudaPS.o link.o cpu_join.o
	$(CC) -o spatialjoin cudaPS.o link.o cpu_join.o  $(LIBB) $(LIBRA) -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/11.2/lib64 -lcudart -lcudadevrt
			
cpu_join.o: cpu_join.cpp
	$(CC) $(CFLAGS) -std=c++11 cpu_join.cpp $(LIBRA) $(LIBB)

cudaPS.o: 
	nvcc -arch=sm_35 -dc cudaPS.cu  
	
link.o:
	nvcc -arch=sm_35 -dlink cudaPS.o -o link.o -lcudadevrt