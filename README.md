# GEMM_TVM_ASM

---
## How to build
**Install required build dependencies:**
* git 
* python3
* LLVM
* TVM (v0.8 Release)

TVM is used to generate scheduler with good performance. Please follow the tutorial https://tvm.apache.org/docs/install/from_source.html to install TVM .   

**Git clone GEMM_TVM_ASM repo**
```bash
git clone https://github.com/wudu98/GEMM_TVM_ASM.git
cd GEMM_TVM_ASM
```

**Test small gemm**
```bash
nohup bash ./small_gemm/benchmark.sh &
```

**Test tvm asm kernel**
```bash
nohup bash ./tvm_asm_kernel/model_tune/tune_resnet50.sh {TUNE_NUM} &
```