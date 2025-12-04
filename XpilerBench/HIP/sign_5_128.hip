__global__ void __launch_bounds__(640)
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

  hipMalloc(&d_A, size * sizeof(float));
  hipMalloc(&d_C, size * sizeof(float));

  hipMemcpy(d_A, A, size * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(640);
  dim3 numBlocks((size + 640 - 1) / 640);

  sign<<<numBlocks, blockSize>>>(d_A, d_C);

  hipMemcpy(C, d_C, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}
