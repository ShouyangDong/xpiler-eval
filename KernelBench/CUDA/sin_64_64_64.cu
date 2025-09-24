

__global__ void __launch_bounds__(256)
    sin(const float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 262144) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int d0, int d1, int d2) {
  float *d_A, *d_C;
  const int total_elements = d0 * d1 * d2;

  cudaMalloc(&d_A, total_elements * sizeof(float));
  cudaMalloc(&d_C, total_elements * sizeof(float));

  cudaMemcpy(d_A, h_A, total_elements * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(256);

  dim3 numBlocks((total_elements + blockSize.x - 1) / blockSize.x);

  sin<<<numBlocks, blockSize>>>(d_A, d_C);
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}