# Makefile for CUDA Slab Allocator project
NVCC       := nvcc
NVCC_FLAGS := -std=c++14 -arch=sm_80 -x=cu --forward-unknown-to-host-compiler

HEADERS := allocator.h log.h typedAlloc.h

all: main.exe paraTest.exe

main.exe: main.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

paraTest.exe: paraTest.cu log.cpp $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(filter-out $(HEADERS),$^) -o $@

run: main.exe paraTest.exe
	./main.exe
	./paraTest.exe

run-main: main.exe
	./main.exe

run-para: paraTest.exe
	./paraTest.exe

clean:
	rm -f main.exe paraTest.exe

.PHONY: all run run-main run-para clean