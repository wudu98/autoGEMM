CXX = clang++
CFLAGS= -march=armv8-a -O3 -std=c++14 -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm
all:
	$(CXX) $(CFLAGS) src/c_file_asm.cpp -o benchmark_kernel
