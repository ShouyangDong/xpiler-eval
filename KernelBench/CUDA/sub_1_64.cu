__global__ void __launch_bounds__(64)
    sub(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 64) {
    C[idx] = A[idx] - B[idx];
  }
}

extern "C" void sub_kernel(float *h_A, float *h_B, float *h_C, int total) {
  float *d_A, *d_B, *d_C;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_B, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(64);
  dim3 numBlocks(1);

  sub<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
