# autoGEMM

---
## How to build
**Install required build dependencies:**
* git 
* python3
* LLVM
* TVM (v0.10 Release)

TVM is used to generate scheduler with near-to-peak performance. Please follow the tutorial https://tvm.apache.org/docs/install/from_source.html to install TVM.

**Git clone autoGEMM repo**
```bash
git clone https://github.com/wudu98/autoGEMM.git
cd autoGEMM
```

**Test experiment**
```bash
bash ./experiment/pipeline_optimization/benchmark.sh
bash ./experiment/RBSA_optimization/benchmark.sh
```

**Test small GEMM**
```bash
nohup bash ./small_gemm/scripts/benchmark.sh &
```
or
```bash
bash ./small_gemm/scripts/run_single_case.sh {M} {N} {K} {UNROLL_K} {NR} {REPEATS}
```

**Test whole framework**
```bash
nohup bash ./tvm_asm_kernel/scripts/model_tune/tune_resnet50.sh {ARCH(mac/linux/a64fx)} {THREADS} {TUNE_STEPS} &
bash ./tvm_asm_kernel/scripts/utils/evaluate.sh {ARCH(mac/linux/a64fx)} {THREADS} {REPEATS} 
```