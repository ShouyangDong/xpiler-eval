#include "hip/hip_runtime.h"


__global__ void __launch_bounds__(1024)
    sin(const float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 131072) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int total_elements) {
  float *d_A, *d_C;

  hipMalloc(&d_A, total_elements * sizeof(float));
  hipMalloc(&d_C, total_elements * sizeof(float));

  hipMemcpy(d_A, h_A, total_elements * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(1024);

  dim3 numBlocks((total_elements + blockSize.x - 1) / blockSize.x);

  sin<<<numBlocks, blockSize>>>(d_A, d_C);
  hipDeviceSynchronize();

  hipMemcpy(h_C, d_C, total_elements * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}