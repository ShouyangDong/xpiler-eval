

__global__ void __launch_bounds__(294)
    sin(float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = threadIdx.x;
  if (idx < 294) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int total) {
  float *d_A, *d_C;

  hipMalloc(&d_A, total * sizeof(float));
  hipMalloc(&d_C, total * sizeof(float));

  hipMemcpy(d_A, h_A, total * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks(1);

  sin<<<numBlocks, blockSize>>>(d_A, d_C);

  hipMemcpy(h_C, d_C, total * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}
