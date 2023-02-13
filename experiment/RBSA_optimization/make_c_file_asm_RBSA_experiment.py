# Generate any shape of asm micro-kernel and ensure correct
# Compile by clang++ -O3
# older version for lastest small gemm

import random
import string
import sys

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
UNROLL_K = int(sys.argv[4])
NR_MAIN = int(sys.argv[5])
repeat = int(sys.argv[6])

RB_strategy = int(sys.argv[7])
# 0 - OpenBLAS strategy
# 1 - LIBXSMM strategy
# 2 - Our RBSA strategy

real_cal_M = M
real_cal_N = N
if RB_strategy == 0:
  M = (M + 5 - 1) // 5 * 5
  N = (N + 16 - 1) // 16 * 16

SIMD_LANE = 4
assert (SIMD_LANE == 4)

RESERVED_REG_NUM = 16

# print('M=%d, N=%d, K=%d' % (M, N, K))

def micro_kernel_loop_asm(LOOP_ID, LAST_K_ID, LINES, COLS, real_lines, real_cols, next_lines, next_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG):
    code_str = ""

    UNROLL_NR = 2
    if COLS % 2 != 0 :
      UNROLL_NR = 1
    if LOOP_ID == LAST_K_ID and WITH_BIAS_FLAG :
      UNROLL_NR = COLS

    A_odd_flag = (LOOP_ID // SIMD_LANE) % 2
    B_odd_flag = ((LOOP_ID * COLS + VEC_REG_B_LEN) // COLS) % 2
    ptr_B_POS = (LOOP_ID * COLS + VEC_REG_B_LEN) % COLS 
    mod_simd_lane_loop_id = LOOP_ID % SIMD_LANE

    vector_scroll_A = [[], []]
    vector_scroll_A[0] = [ vector_id_array_A[i] for i in range(LINES)]
    vector_scroll_A[1] = [ vector_id_array_A[(i+real_lines)%VEC_REG_A_LEN ] for i in range(real_lines) ]

    vector_scroll_B = [(i + LOOP_ID * COLS) % VEC_REG_B_LEN for i in range(COLS)]

    # Initializes the ABC Block pointer
    if REG_BLOCK_TRANS_FLAG and LOOP_ID == LAST_K_ID:
      if REG_BLOCK_TRANS_FLAG == 2 :
        code_str += f"    \"mov     x10, %[A]                 \\n\"\n"
        code_str += f"    \"add     %[B], %[B], #{real_cols * 4}                 \\n\"\n"
        code_str += f"    \"add     %[C], %[C], #{real_cols * 4}                 \\n\"\n"
        code_str += f"    \"mov     x13, %[C]                 \\n\"\n"
      code_str += f"    \"mov     x11, %[B]                   \\n\"\n"
      code_str += f"    \"add     x12, %[B], %[ldb], lsl #2               \\n\"\n" 
      code_str += f"    \"prfm    PLDL1KEEP, [x11, #64]              \\n\"\n"
      code_str += f"    \"prfm    PLDL1KEEP, [x12, #64]              \\n\"\n"
      B_odd_flag = 0

    for i in range(LINES*COLS//UNROLL_NR):
      line = i % LINES
      col = i // LINES
      
      # Main computing
      if FMA_CALCULATE_FLAG :
        if(LOOP_ID == 0 and LOOP_K_BEGIN_FLAG and (not WITH_BIAS_FLAG)):
          for j in range(UNROLL_NR):
            if(line < real_lines and SIMD_LANE*UNROLL_NR*col + SIMD_LANE*j < real_cols):
              code_str += f"    \"fmul    v{line*COLS + col*UNROLL_NR + j}.4s, v{vector_id_array_B[vector_scroll_B[col*UNROLL_NR + j]]}.4s, v{vector_scroll_A[A_odd_flag][line]}.s[{mod_simd_lane_loop_id}]             \\n\"\n"
        else:
          for j in range(UNROLL_NR):
            if(line < real_lines and SIMD_LANE*UNROLL_NR*col + SIMD_LANE*j < real_cols):
              if A_odd_flag == 1 and ((LOOP_ID == LAST_K_ID and not WITH_BIAS_FLAG) or (not LOOP_ID == LAST_K_ID and mod_simd_lane_loop_id == 3)) :
                ori_line = line
                line = (line + VEC_REG_A_LEN % real_lines) % real_lines
              code_str += f"    \"fmla    v{line*COLS + col*UNROLL_NR + j}.4s, v{vector_id_array_B[vector_scroll_B[col*UNROLL_NR + j]]}.4s, v{vector_scroll_A[A_odd_flag][line]}.s[{mod_simd_lane_loop_id}]             \\n\"\n"
              if A_odd_flag == 1 and ((LOOP_ID == LAST_K_ID and not WITH_BIAS_FLAG) or (not LOOP_ID == LAST_K_ID and mod_simd_lane_loop_id == 3)) :
                line = ori_line
               
      # Store C
      if(STORE_C_FLAG and LOOP_ID == LAST_K_ID):
        for j in range(UNROLL_NR):
          if line < real_lines :
            if A_odd_flag == 1 and (not WITH_BIAS_FLAG) :
              ori_line = line
              line = (line + VEC_REG_A_LEN % real_lines) % real_lines
            if(SIMD_LANE*UNROLL_NR*col + SIMD_LANE*(j+1) <= real_cols):
              code_str += f"    \"str     q{line*COLS + col*UNROLL_NR + j}, [x{RESERVED_REG_NUM+line}], #16           \\n\"\n"
            else:
              for k in range(SIMD_LANE*UNROLL_NR*col + SIMD_LANE*j, real_cols):
                code_str += f"    \"st1     {{v{line*COLS + col*UNROLL_NR + j}.s}}[{k%4}], [x{RESERVED_REG_NUM+line}], #4           \\n\"\n"
            if A_odd_flag == 1 and (not WITH_BIAS_FLAG) :
              line = ori_line
      
      if LOOP_K_END_FLAG and LOOP_ID == LAST_K_ID:
        continue

      if not WITH_BIAS_FLAG:
        # Get next block C address
        if (REG_BLOCK_TRANS_FLAG and LOOP_ID == LAST_K_ID and line == LINES - 1 and col == COLS//UNROLL_NR - 1):
          for j in range(next_lines):
            if (j == 0):
              code_str += f"    \"mov     x{RESERVED_REG_NUM}, x13    \\n\"\n"
            elif(j == 1):
              code_str += f"    \"add     x{RESERVED_REG_NUM+1}, x13, x9     \\n\"\n"
            else:
              code_str += f"    \"add     x{RESERVED_REG_NUM+j}, x{RESERVED_REG_NUM+j-2}, x9, lsl #1    \\n\"\n"
      else:
        # Get next block C address
        if (REG_BLOCK_TRANS_FLAG and LOOP_ID == LAST_K_ID):
          if line < next_lines :
            if (line == 0):
              code_str += f"    \"mov     x{RESERVED_REG_NUM}, x13    \\n\"\n"
            elif(line == 1):
              code_str += f"    \"add     x{RESERVED_REG_NUM+1}, x13, x9     \\n\"\n"
            else:
              code_str += f"    \"add     x{RESERVED_REG_NUM+line}, x{RESERVED_REG_NUM+line-2}, x9, lsl #1    \\n\"\n"
        # Load next block C in vector register
        if REG_BLOCK_TRANS_FLAG and LOOP_ID == LAST_K_ID:
          for j in range(UNROLL_NR):
            if(line < next_lines and SIMD_LANE*UNROLL_NR*col + SIMD_LANE*j < next_cols):
              code_str += f"    \"ldr     q{line*COLS + col*UNROLL_NR + j}, [x{RESERVED_REG_NUM+line}, #{(col*UNROLL_NR + j)*16}]           \\n\"\n"

      # Get next block A address
      if (REG_BLOCK_TRANS_FLAG and LOOP_ID == LAST_K_ID and line == 0 and col == 0):
        for j in range(next_lines):
          if (j == 0):
            code_str += f"    \"mov     x{RESERVED_REG_NUM+LINES}, x10    \\n\"\n"
          elif(j == 1):
            code_str += f"    \"add     x{RESERVED_REG_NUM+LINES+1}, x10, x6    \\n\"\n"
          else:
            code_str += f"    \"add     x{RESERVED_REG_NUM+LINES+j}, x{RESERVED_REG_NUM+LINES+j-2}, x6, lsl #1    \\n\"\n"

      # Load next A in vector register
      if not REG_BLOCK_TRANS_FLAG :
        # Sequence load next A
        # The code contains the permutation operation, registers used to scroll A to improve performance
        # Corresponding to the Main computing
        if ((LAST_K_ID == -1 or LOOP_ID < (LAST_K_ID - LAST_K_ID%4)) and line == 0 and col == 0):
          ori_line = line
          for line in range(real_lines):
            if(mod_simd_lane_loop_id == line % 3 and (line >= real_lines - VEC_REG_A_LEN % real_lines or 2 * real_lines <= VEC_REG_A_LEN)):
              if A_odd_flag == 0:
                line = (line + VEC_REG_A_LEN % real_lines) % real_lines
              code_str += f"    \"ldr     q{vector_scroll_A[A_odd_flag^1][line]}, [x{RESERVED_REG_NUM+LINES+line}], #16    \\n\"\n"
          line = ori_line

        if((LAST_K_ID == -1 or LOOP_ID < (LAST_K_ID - LAST_K_ID%4)) and mod_simd_lane_loop_id == 3 and line < real_lines and col == (real_cols+SIMD_LANE-1)//SIMD_LANE//UNROLL_NR - 1):
          if (2 * real_lines > VEC_REG_A_LEN and line < real_lines - VEC_REG_A_LEN % real_lines):
            if A_odd_flag == 0:
              ori_line = line
              line = (line + VEC_REG_A_LEN % real_lines) % real_lines
            code_str += f"    \"ldr     q{vector_scroll_A[A_odd_flag^1][line]}, [x{RESERVED_REG_NUM+LINES+line}], #16    \\n\"\n"
            if A_odd_flag == 0:
              line = ori_line
      else :
        if not WITH_BIAS_FLAG:
          # Load next block A
          if(LOOP_ID == LAST_K_ID and line < next_lines and col == (real_cols+SIMD_LANE-1)//SIMD_LANE//UNROLL_NR - 1) :
            code_str += f"    \"ldr     q{vector_scroll_A[0][line]}, [x{RESERVED_REG_NUM+LINES+line}], #16    \\n\"\n"
        else:
          # Load next block A
          if(LOOP_ID == LAST_K_ID and line < next_lines and col == (real_cols+SIMD_LANE-1)//SIMD_LANE//UNROLL_NR - 1) :
            if A_odd_flag == 0 or line >= real_lines - VEC_REG_A_LEN % real_lines:
              code_str += f"    \"ldr     q{vector_scroll_A[0][line]}, [x{RESERVED_REG_NUM+LINES+line}], #16    \\n\"\n"

      # Sequence Load next B in vector register
      if (line == LINES - 1):
        for j in range(UNROLL_NR):
          if(((not LOOP_ID == LAST_K_ID) or (LOOP_ID == LAST_K_ID and COLS == VEC_REG_B_LEN)) and SIMD_LANE*UNROLL_NR*col + SIMD_LANE*j < real_cols):
            if (LOOP_ID == LAST_K_ID - 1 and UNROLL_NR*col + j >= 2 * COLS - VEC_REG_B_LEN):
              continue
            code_str += f"    \"ldr     q{vector_id_array_B[vector_scroll_B[col*UNROLL_NR + j]]}, [x{register_scroll_B[B_odd_flag]}, #{(ptr_B_POS)*16}]             \\n\"\n"
            # Get next B address
            if ptr_B_POS == COLS - 1:
              ptr_B_POS = 0
              code_str += f"    \"add     x{register_scroll_B[B_odd_flag]}, x{register_scroll_B[B_odd_flag]}, x8              \\n\"\n"
              B_odd_flag ^= 1
            else:
              ptr_B_POS += 1 
    
    # Extra operations ensure that load next block A works correctly
    if REG_BLOCK_TRANS_FLAG and LOOP_ID == LAST_K_ID and WITH_BIAS_FLAG:
      for line in range(next_lines):
        if not (A_odd_flag == 0 or line >= real_lines - VEC_REG_A_LEN % real_lines):
          code_str += f"    \"ldr     q{vector_scroll_A[0][line]}, [x{RESERVED_REG_NUM+LINES+line}], #16    \\n\"\n"
    
    # Extra operations ensure that Load next block B works correctly
    if REG_BLOCK_TRANS_FLAG and LOOP_ID == LAST_K_ID and (not COLS == VEC_REG_B_LEN):
      vector_scroll_B = [i for i in range(VEC_REG_B_LEN)]
      ptr_B_POS = 0
      for j in range(VEC_REG_B_LEN):
        code_str += f"    \"ldr     q{vector_id_array_B[vector_scroll_B[j]]}, [x{register_scroll_B[B_odd_flag]}, #{(ptr_B_POS)*16}]             \\n\"\n"
        if ptr_B_POS == COLS - 1:
          ptr_B_POS = 0
          code_str += f"    \"add     x{register_scroll_B[B_odd_flag]}, x{register_scroll_B[B_odd_flag]}, x8              \\n\"\n"
          B_odd_flag ^= 1
        else:
          ptr_B_POS += 1 

    return code_str

def UNROLL_LOOP_ID(K, UNROLL_K):
  BEGIN_LOOP = 1
  assert (BEGIN_LOOP == 1)
  EDGE_BEGIN_LOOP = 1
  if K % UNROLL_K > SIMD_LANE :
    EDGE_BEGIN_LOOP = (K % UNROLL_K) - (K % SIMD_LANE)
  elif K % UNROLL_K == 0 and UNROLL_K > SIMD_LANE :
    EDGE_BEGIN_LOOP = UNROLL_K - SIMD_LANE
  return BEGIN_LOOP, EDGE_BEGIN_LOOP

def compile_time_for_init_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 1
    FMA_CALCULATE_FLAG = 0
    STORE_C_FLAG = 0
    WITH_BIAS_FLAG = with_bias

    code_str = ""
    code_str += micro_kernel_loop_asm(-1, -1, LINES, COLS, real_lines, real_cols, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str

def compile_time_for_loop_k_begin_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias):
    LOOP_K_BEGIN_FLAG = 1
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 0
    WITH_BIAS_FLAG = with_bias

    MAIN_K_LOOP_BEGIN = 0
    MAIN_K_LOOP_END, _ = UNROLL_LOOP_ID(K, UNROLL_K)

    code_str = f""

    tmp_LINES = real_lines
    cnt = 0
    while(tmp_LINES != 0):
      if (tmp_LINES % 2 != 0):
        if cnt == 0:
          code_str += f"    \"add     x10, x10, x6               \\n\"\n"
          code_str += f"    \"add     x13, x13, x9               \\n\"\n"
        else:
          code_str += f"    \"add     x10, x10, x6, lsl #{cnt}               \\n\"\n"
          code_str += f"    \"add     x13, x13, x9, lsl #{cnt}               \\n\"\n"
      tmp_LINES = tmp_LINES // 2
      cnt += 1

    for LOOP_ID in range(MAIN_K_LOOP_BEGIN, MAIN_K_LOOP_END):
      code_str += micro_kernel_loop_asm(LOOP_ID, -1, LINES, COLS, real_lines, real_cols, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str

def compile_time_for_loop_k_main_body_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 0
    WITH_BIAS_FLAG = 0

    MAIN_K_LOOP_BEGIN, _ = UNROLL_LOOP_ID(K, UNROLL_K)
    MAIN_K_LOOP_END = UNROLL_K + MAIN_K_LOOP_BEGIN

    code_str = f""

    for LOOP_ID in range(MAIN_K_LOOP_BEGIN, MAIN_K_LOOP_END):
      code_str += micro_kernel_loop_asm(LOOP_ID, -1, LINES, COLS, real_lines, real_cols, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str


def compile_time_for_loop_k_remain_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 0
    WITH_BIAS_FLAG = 0

    REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END = UNROLL_LOOP_ID(K, UNROLL_K)

    code_str = f""

    for LOOP_ID in range(REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END):
      code_str += micro_kernel_loop_asm(LOOP_ID, -1, LINES, COLS, real_lines, real_cols, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str
    
    
def compile_time_for_m_dim_micro_kernel_pipeline_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, next_lines, next_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 1
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 1
    WITH_BIAS_FLAG = with_bias
    
    _, REMAIN_K_LOOP_BEGIN = UNROLL_LOOP_ID(K, UNROLL_K)
    REMAIN_K_LOOP_END = UNROLL_K if K % UNROLL_K == 0 else K % UNROLL_K
    
    code_str = f""

    for line in range(real_lines):
      code_str += f"    \"prfm    PSTL1KEEP, [x{RESERVED_REG_NUM+line}, #64]              \\n\"\n"

    for LOOP_ID in range(REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END):
      code_str += micro_kernel_loop_asm(LOOP_ID, REMAIN_K_LOOP_END-1, LINES, COLS, real_lines, real_cols, next_lines, next_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    # When K module UNROLL_K remainder 1, no calculation, direct store
    if (REMAIN_K_LOOP_BEGIN == REMAIN_K_LOOP_END):
      FMA_CALCULATE_FLAG = 0
      code_str += micro_kernel_loop_asm(-1, -1, LINES, COLS, real_lines, real_cols, next_lines, next_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str

def compile_time_for_n_dim_micro_kernel_pipeline_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, next_lines, next_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 2
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 1
    WITH_BIAS_FLAG = with_bias

    _, REMAIN_K_LOOP_BEGIN = UNROLL_LOOP_ID(K, UNROLL_K)
    REMAIN_K_LOOP_END = UNROLL_K if K % UNROLL_K == 0 else K % UNROLL_K
    
    code_str = f""

    for line in range(real_lines):
      code_str += f"    \"prfm    PSTL1KEEP, [x{RESERVED_REG_NUM+line}, #64]              \\n\"\n"

    for LOOP_ID in range(REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END):
      code_str += micro_kernel_loop_asm(LOOP_ID, REMAIN_K_LOOP_END-1, LINES, COLS, real_lines, real_cols, next_lines, next_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    # When K module UNROLL_K remainder 1, no calculation, direct store
    if (REMAIN_K_LOOP_BEGIN == REMAIN_K_LOOP_END):
      FMA_CALCULATE_FLAG = 0
      code_str += micro_kernel_loop_asm(-1, -1, LINES, COLS, real_lines, real_cols, next_lines, next_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str

def compile_time_for_loop_k_end_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 1
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 1
    WITH_BIAS_FLAG = 0

    _, REMAIN_K_LOOP_BEGIN = UNROLL_LOOP_ID(K, UNROLL_K)
    REMAIN_K_LOOP_END = UNROLL_K if K % UNROLL_K == 0 else K % UNROLL_K
    
    code_str = f""

    for line in range(real_lines):
      code_str += f"    \"prfm    PSTL1KEEP, [x{RESERVED_REG_NUM+line}, #64]              \\n\"\n"

    for LOOP_ID in range(REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END):
      code_str += micro_kernel_loop_asm(LOOP_ID, REMAIN_K_LOOP_END-1, LINES, COLS, real_lines, real_cols, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    # When K module UNROLL_K remainder 1, no calculation, direct store
    if (REMAIN_K_LOOP_BEGIN == REMAIN_K_LOOP_END):
      FMA_CALCULATE_FLAG = 0
      code_str += micro_kernel_loop_asm(-1, -1, LINES, COLS, real_lines, real_cols, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str


def m_dim_func_asm(MR_MAIN, MR_MAIN_LOOPS, MR_REMAIN, MR_REMAIN_LOOPS, NR, K, UNROLL_K, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias):
    
    Main_K_loop_flag = (K > UNROLL_K) # The K-dim main operation needs to loop through
    Main_K_loop_times = (K + UNROLL_K - 1) // UNROLL_K
    
    code_str = f""

    if MR_MAIN_LOOPS : # Enter the M-dim main operation
      if MR_MAIN_LOOPS > 1 : # Cyclic M-dim main operation
        code_str += f"    \"mov     x14, #{MR_MAIN_LOOPS}                   \\n\"\n"
        code_str += f"    \"b       1f                                 \\n\"\n"
        code_str += f"  \"2:                                 \\n\"\n"
        code_str += f"    \"subs    x14, x14, #1                            \\n\"\n"
        code_str += compile_time_for_loop_k_remain_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_MAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B)
        code_str += f"    \"beq     3f              \\n\"\n"
        code_str += compile_time_for_m_dim_micro_kernel_pipeline_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_MAIN, real_cols, MR_MAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
        code_str += f"  \"1:                                 \\n\"\n"

      # K-dim main operation
      if Main_K_loop_flag : 
        code_str += f"    \"mov     x15, #{Main_K_loop_times}                   \\n\"\n"
        code_str += f"    \"subs    x15, x15, #1                            \\n\"\n"
        code_str += compile_time_for_loop_k_begin_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_MAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
        code_str += f"    \"b       4f                                 \\n\"\n"
        code_str += f"  \"5:                                 \\n\"\n"
        code_str += compile_time_for_loop_k_main_body_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_MAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B)
        code_str += f"  \"4:                                 \\n\"\n"
        if MR_MAIN_LOOPS > 1 :
          code_str += f"    \"beq     2b                       \\n\"\n"
        else:
          code_str += f"    \"beq     3f                       \\n\"\n"
        code_str += f"    \"subs    x15, x15, #1                            \\n\"\n"
        code_str += f"    \"b       5b                                 \\n\"\n"
      else:
        code_str += compile_time_for_loop_k_begin_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_MAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
        if MR_MAIN_LOOPS > 1 :
          code_str += f"    \"b       2b                       \\n\"\n"

      if MR_MAIN_LOOPS > 1 or Main_K_loop_flag:
        code_str += f"  \"3:                                 \\n\"\n"

      if not MR_MAIN_LOOPS > 1 :
        code_str += compile_time_for_loop_k_remain_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_MAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B)

    if MR_REMAIN_LOOPS : # Enter the M-dim remain operation
      if MR_MAIN_LOOPS : # Cyclic M-dim remain operation
        code_str += compile_time_for_m_dim_micro_kernel_pipeline_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_MAIN, real_cols, MR_REMAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
      if MR_REMAIN_LOOPS > 1 :
        code_str += f"    \"mov     x14, #{MR_REMAIN_LOOPS}                   \\n\"\n"
        code_str += f"    \"b       1f                                 \\n\"\n"
        code_str += f"  \"2:                                 \\n\"\n"
        code_str += f"    \"subs    x14, x14, #1                            \\n\"\n"
        code_str += compile_time_for_loop_k_remain_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_REMAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B)
        code_str += f"    \"beq     3f              \\n\"\n"
        code_str += compile_time_for_m_dim_micro_kernel_pipeline_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_REMAIN, real_cols, MR_REMAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
        code_str += f"  \"1:                                 \\n\"\n"

      # K-dim main operation
      if Main_K_loop_flag : 
        code_str += f"    \"mov     x15, #{Main_K_loop_times}                   \\n\"\n"
        code_str += f"    \"subs    x15, x15, #1                            \\n\"\n"
        code_str += compile_time_for_loop_k_begin_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_REMAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
        code_str += f"    \"b       4f                                 \\n\"\n"
        code_str += f"  \"5:                                 \\n\"\n"
        code_str += compile_time_for_loop_k_main_body_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_REMAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B)
        code_str += f"  \"4:                                 \\n\"\n"
        if MR_REMAIN_LOOPS > 1 :
          code_str += f"    \"beq     2b                       \\n\"\n"
        else:
          code_str += f"    \"beq     3f                       \\n\"\n"
        code_str += f"    \"subs    x15, x15, #1                            \\n\"\n"
        code_str += f"    \"b       5b                                 \\n\"\n"
      else:
        code_str += compile_time_for_loop_k_begin_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_REMAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
        if MR_REMAIN_LOOPS > 1 :
          code_str += f"    \"b       2b                       \\n\"\n"

      if MR_REMAIN_LOOPS > 1 or Main_K_loop_flag:
        code_str += f"  \"3:                                 \\n\"\n"

      if not MR_REMAIN_LOOPS > 1 :
        code_str += compile_time_for_loop_k_remain_func_asm(MR_MAIN, NR, K, UNROLL_K, MR_REMAIN, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B)
        
    return code_str

def n_dim_func_asm(REMAIN_N, K, UNROLL_K, NR, NR_LOOPS, MR_MAIN, MR_MAIN_LOOPS, MR_REMAIN, MR_REMAIN_LOOPS, with_bias) :

    VEC_REG_B_LEN = NR if K <= 16 else max(4, NR)
    if NR == 6 :
      VEC_REG_B_LEN = NR if K <= 32 else 8
    
    vector_id_array_B = []
    for i in range(MR_MAIN*NR, MR_MAIN*NR+VEC_REG_B_LEN):
      vector_id_array_B.append(i)

    VEC_REG_A_LEN = MR_MAIN if K <= 16 else min(32 - MR_MAIN*NR - VEC_REG_B_LEN, 2*MR_MAIN)

    vector_id_array_A = []
    for i in range(MR_MAIN*NR+VEC_REG_B_LEN, MR_MAIN*NR+VEC_REG_B_LEN+VEC_REG_A_LEN):
      vector_id_array_A.append(i)

    register_scroll_B = [11, 12]

    Main_N_flag = (NR_LOOPS > 1) or (SIMD_LANE * NR == REMAIN_N)
    Edge_N_flag = SIMD_LANE * NR * NR_LOOPS > REMAIN_N
    Edge_N = REMAIN_N % (SIMD_LANE * NR)
    if Edge_N_flag :
      NR_LOOPS -= 1
    
    lines_branch_1 = MR_MAIN if MR_MAIN_LOOPS else MR_REMAIN
    lines_branch_2 = MR_MAIN if not MR_REMAIN_LOOPS else MR_REMAIN
    cols_branch_1 = SIMD_LANE*NR if Main_N_flag else Edge_N
    cols_branch_2 = SIMD_LANE*NR if not Edge_N_flag else Edge_N

    code_str = f""
    code_str += compile_time_for_init_func_asm(MR_MAIN, NR, K, UNROLL_K, lines_branch_1, cols_branch_1, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)

    if Main_N_flag : # Enter the N-dim main operation
      if NR_LOOPS > 1 : # Cyclic N-dim main operation
        code_str += f"    \"mov     x7, #{NR_LOOPS}                   \\n\"\n"
        code_str += f"    \"b       6f                                 \\n\"\n"
        code_str += f"  \"0:                                 \\n\"\n"
        code_str += f"    \"subs    x7, x7, #1                            \\n\"\n"
        code_str += f"    \"beq     7f                       \\n\"\n" 
        code_str += compile_time_for_n_dim_micro_kernel_pipeline_func_asm(MR_MAIN, NR, K, UNROLL_K, lines_branch_2, SIMD_LANE*NR, lines_branch_1, SIMD_LANE*NR, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
        code_str += f"  \"6:                                 \\n\"\n"
      
      code_str += m_dim_func_asm(MR_MAIN, MR_MAIN_LOOPS, MR_REMAIN, MR_REMAIN_LOOPS, NR, K, UNROLL_K, SIMD_LANE*NR, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)

      if NR_LOOPS > 1 : 
        code_str += f"    \"b       0b                                 \\n\"\n"
        code_str += f"  \"7:                                 \\n\"\n"

    if Edge_N_flag : # Enter the N-dim remain operation
      if Main_N_flag : # Cyclic N-dim remain operation
        code_str += compile_time_for_n_dim_micro_kernel_pipeline_func_asm(MR_MAIN, NR, K, UNROLL_K, lines_branch_2, SIMD_LANE*NR, lines_branch_1, Edge_N, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)
      code_str += m_dim_func_asm(MR_MAIN, MR_MAIN_LOOPS, MR_REMAIN, MR_REMAIN_LOOPS, NR, K, UNROLL_K, Edge_N, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, with_bias)

    code_str += compile_time_for_loop_k_end_func_asm(MR_MAIN, NR, K, UNROLL_K, lines_branch_2, cols_branch_2, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B)

    return code_str

def NRSA(N, NR_MAIN):
    CEIL_NC = (N + SIMD_LANE - 1) // SIMD_LANE
    NR_REMAIN = CEIL_NC % NR_MAIN
    NR_MAIN_LOOPS = CEIL_NC // NR_MAIN
    NR_REMAIN_LOOPS = 1 if NR_REMAIN else 0

    if RB_strategy == 2:
      if NR_MAIN == 3 :
        if NR_REMAIN == 1 and NR_MAIN_LOOPS >= 1 :
          NR_MAIN_LOOPS -= 1
          NR_REMAIN = 4
        elif NR_REMAIN == 2 and NR_MAIN_LOOPS >= 1 :
          NR_MAIN_LOOPS -= 1
          NR_REMAIN = 5
      elif NR_MAIN == 4 :
        if NR_REMAIN == 1 and NR_MAIN_LOOPS >= 1 :
          NR_MAIN_LOOPS -= 1
          NR_REMAIN = 5
        elif NR_REMAIN == 2 :
          if NR_MAIN_LOOPS >= 2 :
            NR_MAIN_LOOPS -= 2
            NR_REMAIN = 5
            NR_REMAIN_LOOPS = 2
          elif NR_MAIN_LOOPS >= 1 :
            NR_MAIN_LOOPS -= 1
            NR_REMAIN = 6
        elif NR_REMAIN == 3 and NR_MAIN_LOOPS >= 3 :
          NR_MAIN_LOOPS -= 3
          NR_REMAIN = 5
          NR_REMAIN_LOOPS = 3
      elif NR_MAIN == 5 :
        if NR_REMAIN == 1 :
          if NR_MAIN_LOOPS >= 3 :
            NR_MAIN_LOOPS -= 3
            NR_REMAIN = 4
            NR_REMAIN_LOOPS = 4
          elif NR_MAIN_LOOPS >= 1 :
            NR_MAIN_LOOPS -= 1
            NR_REMAIN = 6
        elif NR_REMAIN == 2 and NR_MAIN_LOOPS >= 2 :
          NR_MAIN_LOOPS -= 2
          NR_REMAIN = 4
          NR_REMAIN_LOOPS = 3
        elif NR_REMAIN == 3 and NR_MAIN_LOOPS >= 1 :
          NR_MAIN_LOOPS -= 1
          NR_REMAIN = 4
          NR_REMAIN_LOOPS = 2

    return NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS

def MRSA(M, NR):
    if RB_strategy == 2:
      MR_MAIN = min(6, (32 - max(4, NR)) // (NR + 1))
    else:
      MR_MAIN = 5

    MR_REMAIN = M % MR_MAIN
    MR_MAIN_LOOPS = M // MR_MAIN
    MR_REMAIN_LOOPS = 1 if MR_REMAIN else 0
    if RB_strategy == 2:
      if MR_MAIN == 5 :
        if MR_REMAIN == 1 :
          if MR_MAIN_LOOPS >= 3 :
            MR_MAIN_LOOPS -= 3
            MR_REMAIN = 4
            MR_REMAIN_LOOPS = 4
          elif MR_MAIN_LOOPS >= 1 :
            MR_MAIN_LOOPS -= 1
            MR_REMAIN = 3
            MR_REMAIN_LOOPS = 2
        elif MR_REMAIN == 2 and MR_MAIN_LOOPS >= 2 :
          MR_MAIN_LOOPS -= 2
          MR_REMAIN = 4
          MR_REMAIN_LOOPS = 3
        elif MR_REMAIN == 3 and MR_MAIN_LOOPS >= 1 :
          MR_MAIN_LOOPS -= 1
          MR_REMAIN = 4
          MR_REMAIN_LOOPS = 2
      elif MR_MAIN == 4 :
        if MR_REMAIN == 1 and MR_MAIN_LOOPS >= 2 :
          MR_MAIN_LOOPS -= 2
          MR_REMAIN = 3
          MR_REMAIN_LOOPS = 3
        elif MR_REMAIN == 2 and MR_MAIN_LOOPS >= 1 :
            MR_MAIN_LOOPS -= 1
            MR_REMAIN = 3
            MR_REMAIN_LOOPS = 2
      elif MR_MAIN == 3 and MR_REMAIN == 1 and MR_MAIN_LOOPS >= 1 :
        MR_MAIN_LOOPS -= 1
        MR_REMAIN = 2
        MR_REMAIN_LOOPS = 2
    
    return MR_MAIN, MR_MAIN_LOOPS, MR_REMAIN, MR_REMAIN_LOOPS

def RBSA(M, N, NR_MAIN):
    NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS = NRSA(N, NR_MAIN)
    NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS, NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS = MRSA(M, NR_MAIN) if NR_MAIN_LOOPS else (0,0,0,0)
    NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS, NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS = MRSA(M, NR_REMAIN) if NR_REMAIN_LOOPS else (0,0,0,0)

    return NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS, NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS, NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS, NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS, NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS

def laf_asm_code(M, N, K, lda, ldb, ldc, UNROLL_K = 8, NR_MAIN = 4, with_bias = 0):

    assert (UNROLL_K % (2*SIMD_LANE) == 0)
    assert (NR_MAIN == 4)

    NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS, NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS, NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS, NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS, NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS = RBSA(M, N, NR_MAIN)

    code_str = ""
    code_str += f"""
  asm volatile(
    "prfm    PLDL1KEEP, [%[A], #64]              \\n"
    "prfm    PLDL1KEEP, [%[B], #64]              \\n"
    "lsl	   x6, %[lda], #2              \\n"
    "lsl	   x8, %[ldb], #3              \\n"
    "lsl	   x9, %[ldc], #2              \\n"
    "mov     x10, %[A]                 \\n"
    "mov     x13, %[C]                 \\n"
"""

    if NR_MAIN_LOOPS :
      code_str += n_dim_func_asm(min(N, SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS), K, UNROLL_K, NR_MAIN, NR_MAIN_LOOPS, NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS, NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS, with_bias)

    if NR_REMAIN_LOOPS :
      if NR_MAIN_LOOPS:
        code_str += f"    \"mov     x10, %[A]                 \\n\"\n"
        code_str += f"    \"add     %[B], %[B], #{SIMD_LANE * NR_MAIN * 4}                 \\n\"\n"
        code_str += f"    \"add     %[C], %[C], #{SIMD_LANE * NR_MAIN * 4}                 \\n\"\n"
        code_str += f"    \"mov     x13, %[C]                 \\n\"\n"
      code_str += n_dim_func_asm(N - SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS, K, UNROLL_K, NR_REMAIN, NR_REMAIN_LOOPS, NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS, NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS, with_bias)

    code_str += f"""
    : [A]"=r"(A),
      [B]"=r"(B),
      [C]"=r"(C)
    : "0"(A),
      "1"(B),
      "2"(C),
      [lda]"r"(lda),
      [ldb]"r"(ldb),
      [ldc]"r"(ldc)
    : "cc", "memory" """
    for line in range(RESERVED_REG_NUM - 6 + 2 * max(NR_MAIN_MR_MAIN, NR_REMAIN_MR_MAIN)):
      code_str += f", \"x{6 + line}\""
    code_str += f"\n                      "
    for i in range(32):
      code_str += f", \"v{i}\""
    code_str +=  f"""
  );
"""
    return code_str

def gemm_MxKxN_impl(M, K, N, lda, ldb, ldc, uniq_id):
    
    """Emit C code for gemm impl."""
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
void small_gemm(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc) {{
"""
    cc_code += laf_asm_code(M, N, K, lda, ldb, ldc, UNROLL_K, NR_MAIN, with_bias = 0)
    cc_code += f"""
}}
void small_gemm_with_bias(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc) {{
"""
    cc_code += laf_asm_code(M, N, K, lda, ldb, ldc, UNROLL_K, NR_MAIN, with_bias = 1)
    cc_code += f"""
}}
}}

extern "C" int gemm_{M}x{K}x{N}_{lda}_{ldb}_{ldc}_xsmm_{uniq_id}(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc){{
  laf::small_gemm(A, B, C, lda, ldb, ldc);
  return 0;
}}

extern "C" int gemm_{M}x{K}x{N}_{lda}_{ldb}_{ldc}_xsmm_with_bias_{uniq_id}(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc){{
  laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  return 0;
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
    laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }}

  Timer t;
  for (int i = 0; i < n_loops; ++i) {{
    laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }}

  float latency = t.getTime();
  float gflops = {real_cal_M} * {real_cal_N} * K * 2 / latency * n_loops / 1000000;
  printf("%.2f, ", gflops);

  bool ACC = false;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  laf::small_gemm(A, B, ourC, lda, ldb, ldc);
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
  laf::small_gemm_with_bias(A, B, ourC, lda, ldb, ldc);
  if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, 1e-5, 1e-5)) {{
    int idx = test_utils::diff_index(refC, ourC, M, N, ldc, 1e-5, 1e-5);
    printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\\n",
           M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
  }} else {{
    //printf("1------passed\\n");
  }}

"""
    cc_code += f"""
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
