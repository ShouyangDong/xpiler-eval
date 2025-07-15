__global__ void __launch_bounds__(12)
    softmax(float *__restrict__ A, float *__restrict__ T_softmax_maxelem) {
  if (threadIdx.x < 12) {

    float maxVal = A[threadIdx.x * 5];
    for (int i = 1; i < 5; ++i) {
      if (A[threadIdx.x * 5 + i] > maxVal) {
        maxVal = A[threadIdx.x * 5 + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 5; ++i) {
      T_softmax_maxelem[threadIdx.x * 5 + i] =
          expf(A[threadIdx.x * 5 + i] - maxVal);
      denom += T_softmax_maxelem[threadIdx.x * 5 + i];
    }

    for (int i = 0; i < 5; ++i) {
      T_softmax_maxelem[threadIdx.x * 5 + i] /= denom;
    }
  }
}

extern "C" void softmax_kernel(float *A, float *C, int size1, int size2) {
  float *d_A;
  float *d_C;

  hipMalloc(&d_A, size1 * size2 * sizeof(float));
  hipMalloc(&d_C, size1 * size2 * sizeof(float));

  hipMemcpy(d_A, A, size1 * size2 * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(12);
  dim3 numBlocks((size1 + 12 - 1) / 12);

  softmax<<<numBlocks, blockSize>>>(d_A, d_C);

  hipMemcpy(C, d_C, size1 * size2 * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}
