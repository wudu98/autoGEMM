# CXX = clang++ # The clang++ compiler compiling neon inline assembly does not combine mul and add instructions into fmla instructions
CXX = g++ 
CFLAGS= -march=armv8-a -O3 -std=c++14 -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm
all:
	$(CXX) $(CFLAGS) c_file_asm.cpp -o benchmark_kernel
