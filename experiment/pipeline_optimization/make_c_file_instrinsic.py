# gemm intrinsic implementations
# compile by g++
# Support for arbitrary shape matrices
# optimization not shown
# This is an instance of ARM NEON intrinsic, which could output correct results on a small matrix of arbitrary shape

import random
import string
import sys
import os

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
UNROLL_K = int(sys.argv[4])
NR_MAIN = int(sys.argv[5])
repeat = int(sys.argv[6])
# print('M=%d, N=%d, K=%d' % (M, N, K))

def compile_time_for_unroll_k(M, N, K, lda, ldb, ldc, COLS, LINES, unroll_nums):
    assert(unroll_nums % 4 == 0)
    code_str = ""
    code_str += f"    int k = 0;\n"
    code_str += f"    for (; k + {unroll_nums} <= {K}; k += {unroll_nums}) {{\n"
    for k_unroll in range(0, unroll_nums, 4):
      for unrool_4 in range(4):          
        for i in range(LINES*COLS):
            line = i % LINES
            col = i // LINES
            if (line == 0):
              code_str += f"      vb[{col}] = vld1q_f32(B + k * {ldb} + {k_unroll * ldb +  unrool_4 * ldb + col * 4});\n"
            if unrool_4 == 0 and col == 0:
              code_str += f"      va[{i}] = vld1q_f32(A + k + {k_unroll} + m * {lda} + {i * lda});\n"
            code_str += f"      vc[{line*COLS+col}] = vmlaq_laneq_f32(vc[{line*COLS+col}], vb[{col}], va[{line}], {unrool_4});\n"
    code_str += f"    }}\n"
    if (K % unroll_nums):
      remain = K % unroll_nums
      for k_unroll in range(0, remain, 4):
        unroll_num = 4
        if(remain % 4 != 0 and k_unroll + 4  >= remain):
          unroll_num = remain % 4
        for i in range(LINES):
          code_str += f"    va[{i}] = vld1q_f32(A + k + {k_unroll} + m * {lda} + {i * lda});\n"
        for unrool_4 in range(unroll_num):
          for i in range(LINES*COLS):
              line = i % LINES
              col = i // LINES
              if (line == 0):
                code_str += f"      vb[{col}] = vld1q_f32(B + k * {ldb} + {k_unroll * ldb +  unrool_4 * ldb + col * 4});\n"
              code_str += f"      vc[{line*COLS+col}] = vmlaq_laneq_f32(vc[{line*COLS+col}], vb[{col}], va[{line}], {unrool_4});\n" 
    return code_str

def compile_time_for_set0(M, N, K, lda, ldb, ldc, COLS, LINES):
    code_str = ""
    for i in range(LINES*COLS):
        code_str += f"    vc[{i}] = vdupq_n_f32((float32_t)0);\n"
    return code_str

def compile_time_for_load(M, N, K, lda, ldb, ldc, COLS, LINES):
    code_str = ""
    for i in range(LINES*COLS):
        line = i // COLS
        col = i % COLS
        code_str += f"    vc[{i}] = vld1q_f32(C + m * {ldc} + {line * ldc + col * 4});\n"
    return code_str

def compile_time_for_store(M, N, K, lda, ldb, ldc, COLS, LINES):
    MASK = N % 4
    code_str = ""
    for i in range(LINES*COLS):
        line = i // COLS
        col = i % COLS
        if (col == COLS-1):
          if MASK == 0:
            code_str += f"    vst1q_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}]);\n"
          if MASK == 1:
            code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}], 0);\n"
          if MASK == 2:
            code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}], 0);\n"
            code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4 + 1}, vc[{i}], 1);\n"
          if MASK == 3:
            code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}], 0);\n"
            code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4 + 1}, vc[{i}], 1);\n"
            code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4 + 2}, vc[{i}], 2);\n"
        else:
          code_str += f"    vst1q_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}]);\n"
    return code_str

def gemm_MxKxN_impl(M, K, N, lda, ldb, ldc, uniq_id):
    """Emit C code for gemm impl."""
    NC = NR_MAIN * 4
    UNROLL_NUM = UNROLL_K
    EXPANDED_NC = ((NC-1)//4 + 1) * 4
    COLS = EXPANDED_NC // 4
    LINES = (32 - COLS) // (COLS + 1) 

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
void small_gemm_fixmn(const float *A, const float *B, float *C) {{

  float32x4_t va[{LINES}];
  float32x4_t vb[{COLS}];
  float32x4_t vc[{LINES * COLS}];

  int m = 0;
"""
    if LINES <= M:
      cc_code += f"""
  for (; m + {LINES} <= {M}; m += {LINES}) {{
"""
      cc_code += compile_time_for_set0(M, NC, K, lda, ldb, ldc, COLS, LINES)
      cc_code += compile_time_for_unroll_k(M, NC, K, lda, ldb, ldc, COLS, LINES, UNROLL_NUM)
      cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, LINES)
      cc_code += f"""
  }}
"""
    if M % LINES:
      lines = M % LINES
      cc_code += compile_time_for_set0(M, NC, K, lda, ldb, ldc, COLS, lines)
      cc_code += compile_time_for_unroll_k(M, NC, K, lda, ldb, ldc, COLS, lines,  UNROLL_NUM)
      cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, lines)
    cc_code += f"""
}}
void small_gemm_fixmn_with_bias(const float *A, const float *B, float *C) {{

  float32x4_t va[{LINES}];
  float32x4_t vb[{COLS}];
  float32x4_t vc[{LINES * COLS}];

  int m = 0;
"""
    if LINES <= M:
      cc_code += f"""
  for (; m + {LINES} <= {M}; m += {LINES}) {{
"""
      cc_code += compile_time_for_load(M, NC, K, lda, ldb, ldc, COLS, LINES)
      cc_code += compile_time_for_unroll_k(M, NC, K, lda, ldb, ldc, COLS, LINES, UNROLL_NUM)
      cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, LINES)
      cc_code += f"""
  }}
"""
    if M % LINES:
      lines = M % LINES
      cc_code += compile_time_for_load(M, NC, K, lda, ldb, ldc, COLS, lines)
      cc_code += compile_time_for_unroll_k(M, NC, K, lda, ldb, ldc, COLS, lines,  UNROLL_NUM)
      cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, lines)
    cc_code += f"""
}}
"""
    REMAIN_N = N % NC
    if REMAIN_N:
      EXPANDED_N = ((REMAIN_N-1)//4 + 1) * 4
      COLS = EXPANDED_N // 4
      LINES = (32 - COLS) // (COLS + 1) 
      cc_code += f"""
void small_gemm_fixn(const float *A, const float *B, float *C) {{

  float32x4_t va[{LINES}];
  float32x4_t vb[{COLS}];
  float32x4_t vc[{LINES * COLS}];

  int m = 0;
  for (; m + {LINES} <= {M}; m += {LINES}) {{
"""
      cc_code += compile_time_for_set0(M, REMAIN_N, K, lda, ldb, ldc, COLS, LINES)
      cc_code += compile_time_for_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, COLS, LINES, UNROLL_NUM)
      cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, COLS, LINES)
      cc_code += f"""
  }}
"""
      if M % LINES:
        lines = M % LINES
        cc_code += compile_time_for_set0(M, REMAIN_N, K, lda, ldb, ldc, COLS, lines)
        cc_code += compile_time_for_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, COLS, lines,  UNROLL_NUM)
        cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, COLS, lines)
      cc_code += f"""
}}
void small_gemm_fixn_with_bias(const float *A, const float *B, float *C) {{

  float32x4_t va[{LINES}];
  float32x4_t vb[{COLS}];
  float32x4_t vc[{LINES * COLS}];

  int m = 0;
  for (; m + {LINES} <= {M}; m += {LINES}) {{
"""
      cc_code += compile_time_for_load(M, REMAIN_N, K, lda, ldb, ldc, COLS, LINES)
      cc_code += compile_time_for_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, COLS, LINES, UNROLL_NUM)
      cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, COLS, LINES)
      cc_code += f"""
  }}
"""
      if M % LINES:
        lines = M % LINES
        cc_code += compile_time_for_load(M, REMAIN_N, K, lda, ldb, ldc, COLS, lines)
        cc_code += compile_time_for_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, COLS, lines,  UNROLL_NUM)
        cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, COLS, lines)
      cc_code += f"""
}}
"""
    cc_code += f"""
}}
void small_gemm(const float *A, const float *B, float *C) {{
  int i=0;
  for (; i+{NC}<={N}; i+={NC})
    laf::small_gemm_fixmn(A, B + i, C + i);
  """
    if REMAIN_N:
      cc_code += f"""laf::small_gemm_fixn(A, B + i, C + i);"""
    cc_code += f"""
}}
void small_gemm_with_bias(const float *A, const float *B, float *C) {{
  int i=0;
  for (; i+{NC}<={N}; i+={NC})
    laf::small_gemm_fixmn_with_bias(A, B + i, C + i);
  """
    if REMAIN_N:
      cc_code += f"""laf::small_gemm_fixn_with_bias(A, B + i, C + i);"""
    cc_code += f"""
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
  const int nc = {NC};

  for (int i = 0; i < n_warming; ++i) {{
    small_gemm_with_bias(A, B, C);
  }}

  Timer t;
  for (int i = 0; i < n_loops; ++i) {{
    small_gemm_with_bias(A, B, C);
  }}

  float latency = t.getTime();
  float gflops = M * N * K * 2 / latency * n_loops / 1000000;
  printf("%.2f, ", gflops);
  bool ACC = false;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  small_gemm(A, B, ourC);
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
  small_gemm_with_bias(A, B, ourC);
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