__global__ void gemm(__half *A, __half *B, float *C) {
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  const int c_row_base = blockIdx.y * 16;
  const int c_col_base = blockIdx.x * 16;

  floatx4 d = {0.0f};

  for (int k_step = 0; k_step < 32; k_step += 16) {
    float16x4 a, b;

#pragma unroll
    for (int i = 0; i < 4; ++i) {

      int a_row = c_row_base + threadIdx.x;
      int a_col = k_step + threadIdx.y * 4 + i;
      a[i] = A[a_row * 32 + a_col];

      int b_row = k_step + threadIdx.y * 4 + i;
      int b_col = c_col_base + threadIdx.x;
      b[i] = B[b_row * 1024 + b_col];
    }

    d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int c_row = c_row_base + threadIdx.x;
    int c_col = c_col_base + threadIdx.y * 4 + i;
    if (c_row < 32 && c_col < 1024) {
      C[c_row * 1024 + c_col] = d[i];
    }
  }
}

extern "C" void gemm_kernel(__half *A, __half *B, float *C, int m, int k, int n) {
  __half *d_A;
  __half *d_B;
  float *d_C;

  hipMalloc(&d_A, m * k * sizeof(__half));
  hipMalloc(&d_B, k * n * sizeof(__half));
  hipMalloc(&d_C, m * n * sizeof(float));

  hipMemcpy(d_A, A, m * k * sizeof(__half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, k * n * sizeof(__half), hipMemcpyHostToDevice);

  dim3 blockSize(64);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16);
  gemm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
