__global__ void bmm(__half *A, __half *B, float *C) {
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  const int M = 128, K = 256, N = 512;
  const int LDA = K;
  const int LDB = N;
  const int LDC = N;
  const int BATCH = 4;

  const int batch_id = blockIdx.z;
  const int c_row_base = blockIdx.y * 16;
  const int c_col_base = blockIdx.x * 16;

  float16_t *batch_A = A + batch_id * M * K;
  float16_t *batch_B = B + batch_id * K * N;
  float *batch_C = C + batch_id * M * N;

  floatx4 d = {0.0f};

  for (int k_step = 0; k_step < K; k_step += 16) {
    float16x4 a, b;

#pragma unroll
    for (int i = 0; i < 4; ++i) {

      int a_row = c_row_base + threadIdx.x;
      int a_col = k_step + threadIdx.y * 4 + i;
      a[i] = batch_A[a_row * LDA + a_col];

      int b_row = k_step + threadIdx.y * 4 + i;
      int b_col = c_col_base + threadIdx.x;
      b[i] = batch_B[b_row * LDB + b_col];
    }

    d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int c_row = c_row_base + threadIdx.x;
    int c_col = c_col_base + threadIdx.y * 4 + i;
    if (c_row < M && c_col < N) {
      batch_C[c_row * LDC + c_col] = d[i];
    }
  }
}

extern "C" void bmm_kernel(__half *A, __half *B, float *C, int b, int m, int k,
                           int n) {
  __half *d_A;
  __half *d_B;
  float *d_C;

  hipMalloc(&d_A, b * m * k * sizeof(__half));
  hipMalloc(&d_B, b * k * n * sizeof(__half));
  hipMalloc(&d_C, b * m * n * sizeof(float));

  hipMemcpy(d_A, A, b * m * k * sizeof(__half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, b * k * n * sizeof(__half), hipMemcpyHostToDevice);

  dim3 blockSize(64);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16, 4);
  bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, b * m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
