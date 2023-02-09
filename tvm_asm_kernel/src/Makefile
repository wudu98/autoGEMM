CXX = clang++
CFLAGS= -std=c++17 -I$(TVM_HOME)/include -I$(TVM_HOME)/3rdparty/dlpack/include -I$(TVM_HOME)/3rdparty/dmlc-core/include -L${TVM_HOME}/build -ltvm -ldl -pthread -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm -Wno-macro-redefined  
all:
	$(CXX) $(CFLAGS) benchmark.cpp -o benchmark_kernel
