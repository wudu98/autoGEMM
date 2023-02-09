#ifndef _TEST_H_
#define _TEST_H_
#include <cstdlib>
#include <cmath>

class test_utils {
public:
  // A: M x K; B: K x N; C: M x N;
  static void gemm_ref(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool ACC) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        // C[i,j] = SUM(A[i,k] * B[k,j])
        float sum = ACC ? C[i * ldc + j] : 0;
        for (int k = 0; k < K; ++k) {
          sum += A[i * lda + k] * B[k * ldb + j];   
        }
        C[i * ldc + j] = sum;
      }
    }
  }
  
  static bool is_same_matrix(const float *C1, const float *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if ((fabs(C1[i * ldc + j] - C2[i * ldc + j]) > atol) && (fabs(C1[i * ldc + j] - C2[i * ldc + j])/fabs(C1[i * ldc + j]) > rtol)) {
          return false;
        }
      }
    }
    return true;
  }

  static int diff_index(const float *C1, const float *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if ((fabs(C1[i * ldc + j] - C2[i * ldc + j]) > atol) && (fabs(C1[i * ldc + j] - C2[i * ldc + j])/fabs(C1[i * ldc + j]) > rtol)) {
          return i * ldc + j;
        }
      }
    }
    return -1;
  }

  static void init(float *buf, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = 1.0f * rand() / RAND_MAX;
        //buf[i] = 1.0f;
    }
  }

};
#endif