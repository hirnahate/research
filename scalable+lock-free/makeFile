# Makefile for CUDA Slab Allocator project
NVCC       := nvcc
NVCC_FLAGS := -std=c++14 -arch=sm_80 -x=cu --forward-unknown-to-host-compiler

all: main paraTest

main.exe: main.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

paraTest.exe: paraTest.cu log.cpp
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

run: main.exe paraTest.exe
	./main.exe
	./paraTest.exe

run-main: main.exe
	./main.exe

run-para: paraTest.exe
	./paraTest.exe


clean:
	rm -f main.exe paraTest.exe