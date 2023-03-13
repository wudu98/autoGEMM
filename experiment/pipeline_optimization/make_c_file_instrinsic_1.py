# gemm intrinsic implementations
# compile by g++
# corresponding to optimization listing 1
# corresponding to figure a) 

import random
import string
import sys
import os

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
NR_MAIN = 4
repeat = int(sys.argv[4])
# print('M=%d, N=%d, K=%d' % (M, N, K))

def compile_time_for_unroll_k(M, N, K, ROWS, COLS):
    code_str = ""
    code_str += f"    for (int k = 0; k < {K}; k++) {{\n"     
    for row in range (ROWS) :
      code_str += f"      va[{row}] = vld1q_f32(A + {row} * lda + k);\n"
    for col in range (COLS) :
      code_str += f"      vb[{col}] = vld1q_f32(B + k * ldb + {col} * 4);\n"
    for row in range (ROWS) :
      for col in range (COLS) :
        code_str += f"      vc[{row*COLS+col}] = vmlaq_laneq_f32(vc[{row*COLS+col}], vb[{col}], va[{row}], {0});\n"
    code_str += f"    }}\n"

    return code_str

def compile_time_for_set0(M, N, K, ROWS, COLS):
    code_str = ""
    for row in range (ROWS) :
      for col in range (COLS) :
        code_str += f"    vc[{row*COLS+col}] = vdupq_n_f32((float32_t)0);\n"
    return code_str

def compile_time_for_load(M, N, K, ROWS, COLS):
    code_str = ""
    for row in range (ROWS) :
      for col in range (COLS) :
        code_str += f"    vc[{row*COLS+col}] = vld1q_f32(C + {row} * ldc + {col} * 4);\n"
    return code_str

def compile_time_for_store(M, N, K, ROWS, COLS):
    code_str = ""
    for row in range (ROWS) :
      for col in range (COLS) :
        code_str += f"    vst1q_f32(C + {row} * ldc + {col} * 4, vc[{row*COLS+col}]);\n"
    return code_str

def gemm_MxKxN_impl(M, K, N, lda, ldb, ldc, uniq_id):
    """Emit C code for gemm impl."""
    NR = NR_MAIN * 4
    VEC_NR = NR_MAIN
    MR = (30 - VEC_NR) // (VEC_NR + 1) 

    cc_code = f"""
#ifndef __SGEMM_KERNEL_H
#define __SGEMM_KERNEL_H
#endif
#include <cmath>
#include <cstring>
#include <cassert>
#include <arm_neon.h>
#include <cstdlib>
#include <cstdio>
#include "test.h"
#include "timer.h"

namespace laf {{
void small_gemm_fixmn(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc) {{

  float32x4_t va[{MR}];
  float32x4_t vb[{VEC_NR}];
  float32x4_t vc[{MR * VEC_NR}];
"""
    cc_code += compile_time_for_set0(M, N, K, MR, VEC_NR)
    cc_code += compile_time_for_unroll_k(M, N, K, MR, VEC_NR)
    cc_code += compile_time_for_store(M, N, K, MR, VEC_NR)
    cc_code += f"""
}}
void small_gemm_fixmn_with_bias(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc) {{

  float32x4_t va[{MR}];
  float32x4_t vb[{VEC_NR}];
  float32x4_t vc[{MR * VEC_NR}];
"""
    cc_code += compile_time_for_load(M, N, K, MR, VEC_NR)
    cc_code += compile_time_for_unroll_k(M, N, K, MR, VEC_NR)
    cc_code += compile_time_for_store(M, N, K, MR, VEC_NR)
    cc_code += f"""
  }}
}} 

void small_gemm(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc) {{
  for (int i=0; i<{N}; i+={NR})
    for (int j=0; j<{M}; j+={MR})
      laf::small_gemm_fixmn(A + j * lda, B + i, C + j * ldc + i, lda, ldb, ldc);
}}
void small_gemm_with_bias(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc) {{
  for (int i=0; i<{N}; i+={NR})
    for (int j=0; j<{M}; j+={MR})
    laf::small_gemm_fixmn_with_bias(A + j * lda, B + i, C + j * ldc + i, lda, ldb, ldc);
}}

void* _mm_malloc(size_t align, size_t sz)
{{
  void *ptr;
  int alloc_result = posix_memalign(&ptr, align, sz);
  if(alloc_result != 0)
  {{
    return NULL;
  }}
  return ptr;
}}

int main() {{
  #define M {M}
  #define N {N}
  #define K {K}

  #define lda {lda}
  #define ldb {ldb}
  #define ldc {ldc}

  float *A = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
  float *B = static_cast<float*>(_mm_malloc(64, K * ldb * sizeof(float)));
  float *C = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
  float *refC = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
  float *ourC = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));

  test_utils::init(A, M * lda);
  test_utils::init(B, K * ldb);
  test_utils::init(C, M * ldc);

  int n_warming = 20;
  int n_loops = {repeat};

  for (int i = 0; i < n_warming; ++i) {{
    small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }}

  Timer t;
  for (int i = 0; i < n_loops; ++i) {{
    small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }}

  float latency = t.getTime();
  float gflops = M * N * K * 2 / latency * n_loops / 1000000;
  printf("%.2f, ", gflops);
  bool ACC = false;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  small_gemm(A, B, ourC, lda, ldb, ldc);
  if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, 1e-5, 1e-5)) {{
    int idx = test_utils::diff_index(refC, ourC, M, N, ldc, 1e-5, 1e-5);
    printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\\n",
           M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
  }} else {{
    //printf("0------passed\\n");
  }}
  for (int i = 0; i < M; ++i) {{
    for (int j = 0; j < N; ++j) {{
      float c = 10.0f * rand() / RAND_MAX;
      refC[i * ldc + j] = c;
      ourC[i * ldc + j] = c;
    }}
  }}
  ACC = true;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  small_gemm_with_bias(A, B, ourC, lda, ldb, ldc);
  if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, 1e-5, 1e-5)) {{
    int idx = test_utils::diff_index(refC, ourC, M, N, ldc, 1e-5, 1e-5);
    printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\\n",
           M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
  }} else {{
    //printf("1------passed\\n");
  }}
  free(A);
  A=NULL;
  free(B);
  B=NULL;
  free(C);
  C=NULL;
  free(refC);
  refC=NULL;
  free(ourC);
  ourC=NULL;
}}
    """
    return cc_code

UNIQ_ID_LEN = 8
uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))
f = open('c_file_asm.cpp','w')
f.write(gemm_MxKxN_impl(M, K, N, K, N, N, uniq_id))
f.close()