#include "hip/hip_runtime.h"
__global__ void gemv(float *A, float *x, float *y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < 112) {
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
      sum += A[row * 128 + i] * x[i];
    }
    y[row] = sum;
  }
}

extern "C" void gemv_kernel(float *A, float *x, float *y, int m, int n) {
  float *d_A, *d_x, *d_y;

  hipMalloc(&d_A, m * n * sizeof(float));
  hipMalloc(&d_x, n * sizeof(float));
  hipMalloc(&d_y, m * sizeof(float));

  hipMemcpy(d_A, A, m * n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

  int blockSize = 112;
  int numBlocks = (m + blockSize - 1) / blockSize;

  gemv<<<numBlocks, blockSize>>>(d_A, d_x, d_y);

  hipMemcpy(y, d_y, m * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
}
