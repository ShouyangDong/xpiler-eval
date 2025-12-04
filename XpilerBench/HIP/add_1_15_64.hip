__global__ void __launch_bounds__(960)
    add(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  T_add[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
}

extern "C" void add_kernel(float *A, float *B, float *C, int size) {
  float *d_A;
  float *d_B;
  float *d_C;

  hipMalloc(&d_A, size * sizeof(float));
  hipMalloc(&d_B, size * sizeof(float));
  hipMalloc(&d_C, size * sizeof(float));

  hipMemcpy(d_A, A, size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, size * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(960);
  dim3 numBlocks((size + 960 - 1) / 960);

  add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
