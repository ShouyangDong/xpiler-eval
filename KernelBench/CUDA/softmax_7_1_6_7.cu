__global__ void __launch_bounds__(42)
    softmax(float *__restrict__ A, float *__restrict__ T_softmax_norm) {
  if (threadIdx.x < 42) {
    int rowStart = threadIdx.x * 7;

    float maxVal = A[rowStart];
    for (int i = 1; i < 7; ++i) {
      if (A[rowStart + i] > maxVal) {
        maxVal = A[rowStart + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 7; ++i) {
      T_softmax_norm[rowStart + i] = expf(A[rowStart + i] - maxVal);
      denom += T_softmax_norm[rowStart + i];
    }

    for (int i = 0; i < 7; ++i) {
      T_softmax_norm[rowStart + i] /= denom;
    }
  }
}

extern "C" void softmax_kernel(float *A, float *C, int size1, int size2) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size1 * size2 * sizeof(float));
  cudaMalloc(&d_C, size1 * size2 * sizeof(float));

  cudaMemcpy(d_A, A, size1 * size2 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(42);
  dim3 numBlocks((size1 + 42 - 1) / 42);

  softmax<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(C, d_C, size1 * size2 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
