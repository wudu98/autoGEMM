#ifndef __TVMGEMM_UTIL_H_
#define __TVMGEMM_UTIL_H_

#include <cstdio>
#include <cstdlib>

#include "dlpack/dlpack.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"

#include "kernel_params_list.hpp"
#include "./test.h"
#include "./timer.h"

void* _mm_malloc(size_t align, size_t sz)
{
  void *ptr;
  int alloc_result = posix_memalign(&ptr, align, sz);
  if(alloc_result != 0)
  {
    return NULL;
  }
  return ptr;
}


int main(int argc, char* argv[]) {
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int nc, kc, padding_size;
    int repeat = atoi(argv[4]);

    KernelParams::CreateList();
    for(auto it = KernelParams::params_list.begin(); it != KernelParams::params_list.end(); it++){
      if (it->M == M && it->N == N && it->K == K) {
        nc = it->nc;
        kc = it->kc;
        padding_size = it->padding_size;
        break;
      }
    }  

    int nc_ceil = ((nc - 1) / padding_size + 1) * padding_size;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    float *A = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
    float *B = static_cast<float*>(_mm_malloc(64, K * ldb * sizeof(float)));
    float *packedB = static_cast<float*>(_mm_malloc(64, K * (N/nc) * nc_ceil * sizeof(float)));
    float *C = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
    float *refC = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
    float *ourC = C;

    test_utils::init(A,M*lda);
    test_utils::init(B,K*ldb);
    test_utils::init(C,M*ldc);

    string base_name = "GEMM_" + std::to_string(M) + "X" + std::to_string(N) + "X" + std::to_string(K);
    string mod_name = base_name + "_kernel.so";
    string func_name = "OP_" + base_name;
    string pack_func_name = func_name + "_packB";

    tvm::runtime::Module mod_tvmlib = tvm::runtime::Module::LoadFromFile("../build/library/" + mod_name);
    // tvm::runtime::Module mod_tvmlib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();

    tvm::runtime::PackedFunc pack_func = mod_tvmlib.GetFunction(pack_func_name);
    tvm::runtime::PackedFunc func = mod_tvmlib.GetFunction(func_name);

    DLTensor*   tvm_A;
    DLTensor*   tvm_B;
    DLTensor*   tvm_packedB;
    DLTensor*   tvm_C;

    const int64_t A_shape[2] = {M, K};
    const int64_t B_shape[2] = {K, N};
    const int64_t packedB_shape[4] = {K / kc, N / nc, kc, nc_ceil};
    const int64_t C_shape[2] = {M, N};

    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;

    TVMArrayAlloc(A_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_A);
    TVMArrayAlloc(B_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_B);
    TVMArrayAlloc(packedB_shape, 4, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_packedB);
    TVMArrayAlloc(C_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_C);

    tvm_A->data = A;
    tvm_B->data = B;
    tvm_packedB->data = packedB;
    tvm_C->data = C;

    double alpha = 1.0;
    double beta = 0.0;
    int n_warming = 20;
    int n_loops = repeat;

    pack_func(tvm_B, tvm_packedB);

    for (int i = 0; i < n_warming; ++i) {  
        func(tvm_A, tvm_packedB, tvm_C);
    }

    Timer t_1;
    for (int i = 0; i < n_loops; ++i) {
        func(tvm_A, tvm_packedB, tvm_C);
    }

    float latency = t_1.getTime();
    float gflops = M * N * K / latency / 1000000 * n_loops * 2;
    printf("offline, M: %d, N: %d, K: %d, perf: %.2f gflops, latency: %.6f ms\n", M, N, K, gflops, latency / n_loops);

    Timer t_2;
    for (int i = 0; i < n_loops; ++i) {
        pack_func(tvm_B, tvm_packedB);
        func(tvm_A, tvm_packedB, tvm_C);
    }
    
    latency = t_2.getTime();
    gflops = M * N * K / latency / 1000000 * n_loops * 2;
    printf("online, M: %d, N: %d, K: %d, perf: %.2f gflops, latency: %.6f ms\n", M, N, K, gflops, latency / n_loops);

    bool ACC = false;
    test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
    pack_func(tvm_B, tvm_packedB);
    func(tvm_A, tvm_packedB, tvm_C);
    if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, 1e-5, 1e-5)) {
      int idx = test_utils::diff_index(refC, ourC, M, N, ldc, 1e-5, 1e-5);
      printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\n",
            M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
    } else {
      // printf("0------passed\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}

#endif