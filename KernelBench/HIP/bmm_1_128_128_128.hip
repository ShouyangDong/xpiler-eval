__global__ void bmm(__half *A, __half *B, float *C) {
  using float16x4 = __attribute__((__vector_size__(4 * sizeof(__half)))) __half;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  floatx4 d = {0.0f};

  int c_row_base = blockIdx.x * 16;
  int c_col_base = blockIdx.y * 16;

  for (int k_step = 0; k_step < 128; k_step += 16) {
    float16x4 a, b;

    for (int i = 0; i < 4; ++i) {

      int a_row = c_row_base + threadIdx.x;
      int a_col = k_step + threadIdx.y * 4 + i;
      int a_idx = a_row * 128 + a_col;
      a[i] = A[a_idx];

      int b_row = k_step + threadIdx.y * 4 + i;
      int b_col = c_col_base + threadIdx.x;
      int b_idx = b_row * 128 + b_col;
      b[i] = B[b_idx];
    }

    d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  }

  for (int i = 0; i < 4; ++i) {
    int c_row = c_row_base + threadIdx.x;
    int c_col = c_col_base + threadIdx.y * 4 + i;
    C[c_row * 128 + c_col] = d[i];
  }
}

extern "C" void bmm_kernel(__half *A, __half *B, float *C, int b, int m, int k,
                           int n) {
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

  bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
