#include "hip/hip_runtime.h"
__global__ void __launch_bounds__(1024)
    relu(float *__restrict__ A, float *__restrict__ compute) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1125) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(
        A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" void relu_kernel(float *A, float *C, int size) {
  float *d_A;
  float *d_C;

  hipMalloc(&d_A, size * sizeof(float));
  hipMalloc(&d_C, size * sizeof(float));

  hipMemcpy(d_A, A, size * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  relu<<<numBlocks, blockSize>>>(d_A, d_C);

  hipMemcpy(C, d_C, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}
