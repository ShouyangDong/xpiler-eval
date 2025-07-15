__global__ void __launch_bounds__(36)
    softmax(float *__restrict__ A, float *__restrict__ T_softmax_exp) {
  if (threadIdx.x < 36) {

    float maxVal = A[threadIdx.x * 128];
    for (int i = 1; i < 128; ++i) {
      if (A[threadIdx.x * 128 + i] > maxVal) {
        maxVal = A[threadIdx.x * 128 + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 128; ++i) {
      T_softmax_exp[threadIdx.x * 128 + i] =
          expf(A[threadIdx.x * 128 + i] - maxVal);
      denom += T_softmax_exp[threadIdx.x * 128 + i];
    }

    for (int i = 0; i < 128; ++i) {
      T_softmax_exp[threadIdx.x * 128 + i] /= denom;
    }
  }
}

extern "C" void softmax_kernel(float *A, float *C, int size1, int size2) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size1 * size2 * sizeof(float));
  cudaMalloc(&d_C, size1 * size2 * sizeof(float));

  cudaMemcpy(d_A, A, size1 * size2 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(36);
  dim3 numBlocks((size1 + 36 - 1) / 36);

  softmax<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(C, d_C, size1 * size2 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
