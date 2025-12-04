__global__ void __launch_bounds__(1024)
    sign(float *__restrict__ A, float *__restrict__ T_sign) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4608) {
    T_sign[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
        ((0.000000e+00f < A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))])
             ? 1.000000e+00f
             : ((A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] <
                 0.000000e+00f)
                    ? -1.000000e+00f
                    : 0.000000e+00f));
  }
}

extern "C" void sign_kernel(float *A, float *C, int size) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  sign<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
