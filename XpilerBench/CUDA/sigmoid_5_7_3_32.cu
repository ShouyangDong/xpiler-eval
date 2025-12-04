__global__ void __launch_bounds__(1024)
    sigmoid(float *__restrict__ A, float *__restrict__ compute) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 3360) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
        (1.000000e+00f /
         (1.000000e+00f +
          __expf((0.000000e+00f -
                  A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]))));
  }
}

extern "C" void sigmoid_kernel(float *A, float *C, int size) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  sigmoid<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
