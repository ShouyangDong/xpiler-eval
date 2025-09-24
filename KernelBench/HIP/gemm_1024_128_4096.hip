__global__ void gemm(half *A, half *B, float *C) {
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  const int M = 1024, K = 128, N = 4096;
  const int LDA = K;
  const int LDB = N;
  const int LDC = N;
  const int TILE = 16;

  const int c_row = blockIdx.y * TILE + threadIdx.x;
  const int c_col = blockIdx.x * TILE + threadIdx.y * 4;

  floatx4 d = {0.0f};
  float16x4 a, b;

  for (int k_base = 0; k_base < K; k_base += TILE) {

#pragma unroll
    for (int i = 0; i < 4; ++i) {

      int a_col = k_base + threadIdx.y * 4 + i;
      a[i] = A[c_row * LDA + a_col];

      int b_row = k_base + threadIdx.y * 4 + i;
      b[i] = B[b_row * LDB + c_col];
    }

    d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    C[c_row * LDC + c_col + i] = d[i];
  }
}

extern "C" void gemm_kernel(half *A, half *B, float *C, int m, int k, int n) {
  half *d_A;
  half *d_B;
  float *d_C;

  hipMalloc(&d_A, m * k * sizeof(half));
  hipMalloc(&d_B, k * n * sizeof(half));
  hipMalloc(&d_C, m * n * sizeof(float));

  hipMemcpy(d_A, A, m * k * sizeof(half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, k * n * sizeof(half), hipMemcpyHostToDevice);

  dim3 blockSize(64);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16);
  gemm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
