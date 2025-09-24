#include <cuda_runtime.h>
#include <iostream>

__global__ void __launch_bounds__(256)
    sub_2x16x1024(const float *__restrict__ A, const float *__restrict__ B,
                  float *__restrict__ C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 32768) {
    C[idx] = A[idx] - B[idx];
  }
}

extern "C" void sub_kernel(float *h_A, float *h_B, float *h_C) {
  float *d_A, *d_B, *d_C;
  const int total = 2 * 16 * 1024;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_B, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 numBlocks((total + 255) / 256);

  sub_2x16x1024<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
    return;
  }

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
