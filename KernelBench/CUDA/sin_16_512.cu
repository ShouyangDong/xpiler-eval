

__global__ void __launch_bounds__(1024)
    sin(const float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 8192) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int total_elements) {
  float *d_A, *d_C;

  cudaMalloc(&d_A, total_elements * sizeof(float));
  cudaMalloc(&d_C, total_elements * sizeof(float));

  cudaMemcpy(d_A, h_A, total_elements * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((total_elements + blockSize.x - 1) / blockSize.x);

  sin<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
