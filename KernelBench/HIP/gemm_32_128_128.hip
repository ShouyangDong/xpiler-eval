__global__ void gemm(half *A, half *B, float *C) {
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  const int M = 32, K = 128, N = 128;
  const int LDA = K;
  const int LDB = N;
  const int LDC = N;

  const int c_row_base = blockIdx.y * 16;
  const int c_col_base = blockIdx.x * 16;

  floatx4 d = {0.0f};

  for (int k_step = 0; k_step < K; k_step += 16) {
    float16x4 a, b;

#pragma unroll
    for (int i = 0; i < 4; ++i) {

      const int a_row = c_row_base + threadIdx.x;
      const int a_col = k_step + threadIdx.y * 4 + i;
      a[i] = (a_row < M && a_col < K) ? A[a_row * LDA + a_col] : 0.0h;

      const int b_row = k_step + threadIdx.y * 4 + i;
      const int b_col = c_col_base + threadIdx.x;
      b[i] = (b_row < K && b_col < N) ? B[b_row * LDB + b_col] : 0.0h;
    }

    d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c_row = c_row_base + threadIdx.x;
    const int c_col = c_col_base + threadIdx.y * 4 + i;
    if (c_row < M && c_col < N) {
      C[c_row * LDC + c_col] = d[i];
    }
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
