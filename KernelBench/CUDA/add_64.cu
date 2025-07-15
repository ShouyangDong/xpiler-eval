__global__ void __launch_bounds__(64)
    add(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  T_add[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
}

extern "C" void add_kernel(float *A, float *B, float *C, int size) {
  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_B, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(64);
  dim3 numBlocks((size + 64 - 1) / 64);

  add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
