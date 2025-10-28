#include "hip/hip_runtime.h"

__device__ float geluf(float x);

__device__ float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

__global__ void __launch_bounds__(1024)
    gelu(float *__restrict__ A, float *__restrict__ compute) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 3360) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
        geluf(A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" void gelu_kernel(float *A, float *C, int size) {
  float *d_A;
  float *d_C;

  hipMalloc(&d_A, size * sizeof(float));
  hipMalloc(&d_C, size * sizeof(float));

  hipMemcpy(d_A, A, size * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  gelu<<<numBlocks, blockSize>>>(d_A, d_C);

  hipMemcpy(C, d_C, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}
