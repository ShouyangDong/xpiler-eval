__global__ void __launch_bounds__(294)
    sign(float *__restrict__ A, float *__restrict__ T_sign) {
  T_sign[((int)threadIdx.x)] =
      ((0.000000e+00f < A[((int)threadIdx.x)])
           ? 1.000000e+00f
           : ((A[((int)threadIdx.x)] < 0.000000e+00f) ? -1.000000e+00f
                                                      : 0.000000e+00f));
}

extern "C" void sign_kernel(float *A, float *C, int size) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks((size + 294 - 1) / 294);

  sign<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
