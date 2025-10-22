#include "hip/hip_runtime.h"
__global__ void __launch_bounds__(105)
    softmax(float *__restrict__ A, float *__restrict__ T_softmax_maxelem) {
  if (threadIdx.x < 105) {

    float maxVal = A[threadIdx.x * 32];
    for (int i = 1; i < 32; ++i) {
      if (A[threadIdx.x * 32 + i] > maxVal) {
        maxVal = A[threadIdx.x * 32 + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 32; ++i) {
      T_softmax_maxelem[threadIdx.x * 32 + i] =
          expf(A[threadIdx.x * 32 + i] - maxVal);
      denom += T_softmax_maxelem[threadIdx.x * 32 + i];
    }

    for (int i = 0; i < 32; ++i) {
      T_softmax_maxelem[threadIdx.x * 32 + i] /= denom;
    }
  }
}

extern "C" void softmax_kernel(float *A, float *C, int size1, int size2) {
  float *d_A;
  float *d_C;

  hipMalloc(&d_A, size1 * size2 * sizeof(float));
  hipMalloc(&d_C, size1 * size2 * sizeof(float));

  hipMemcpy(d_A, A, size1 * size2 * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(105);
  dim3 numBlocks((size1 + 105 - 1) / 105);

  softmax<<<numBlocks, blockSize>>>(d_A, d_C);

  hipMemcpy(C, d_C, size1 * size2 * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}
