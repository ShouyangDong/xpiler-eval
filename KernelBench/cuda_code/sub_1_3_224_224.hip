#include "hip/hip_runtime.h"


__global__ void __launch_bounds__(1024)
    sub(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 150528) {
    C[idx] = A[idx] - B[idx];
  }
}

extern "C" void sub_kernel(float *h_A, float *h_B, float *h_C) {
  float *d_A, *d_B, *d_C;
  const int total = 1 * 3 * 224 * 224;

  hipMalloc(&d_A, total * sizeof(float));
  hipMalloc(&d_B, total * sizeof(float));
  hipMalloc(&d_C, total * sizeof(float));

  hipMemcpy(d_A, h_A, total * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B, total * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((total + blockSize.x - 1) / blockSize.x);

  sub<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipDeviceSynchronize();

  hipMemcpy(h_C, d_C, total * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
