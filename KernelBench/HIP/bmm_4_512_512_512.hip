__global__ void bmm(half *A, half *B, float *C) {
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  const int LDA = 512;
  const int LDB = 512;
  const int LDC = 512;
  const int BATCH_SIZE = 4;

  int batch_id = blockIdx.z;
  int c_row_base = (blockIdx.x % 8) * 16;
  int c_col_base = (blockIdx.x / 8) * 16;

  float16_t *batch_A = A + batch_id * 512 * 512;
  float16_t *batch_B = B + batch_id * 512 * 512;
  float *batch_C = C + batch_id * 512 * 512;

  floatx4 d = {0.0f};

  for (int k_step = 0; k_step < 512; k_step += 16) {
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
    batch_C[c_row * LDC + c_col] = d[i];
  }
}

extern "C" void bmm_kernel(half *A, half *B, float *C, int b, int m, int k,
                           int n) {
  half *d_A;
  half *d_B;
  float *d_C;

  hipMalloc(&d_A, b * m * k * sizeof(half));
  hipMalloc(&d_B, b * k * n * sizeof(half));
  hipMalloc(&d_C, b * m * n * sizeof(float));

  hipMemcpy(d_A, A, b * m * k * sizeof(half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, b * k * n * sizeof(half), hipMemcpyHostToDevice);

  dim3 blockSize(64, 1, 4);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16);
  bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, b * m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
