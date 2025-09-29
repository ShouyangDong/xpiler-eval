#define M 16
#define N 16
#define K 4

#define B_outer 4
#define M_outer 512
#define N_outer 512
#define K_outer 512

__global__ void bmm(const float *A, const float *B, float *D) {
  int m_base = blockIdx.x * M;
  int n_base = blockIdx.y * N;
  using float4 = __attribute__((__vector_size__(K * sizeof(float)))) float;
  float4 dmn = {0};

  for (int kk = 0; kk < K_outer; kk += K) {
    int a_row = m_base + threadIdx.x;
    int a_col = kk + threadIdx.y;
    float amk = A[blockIdx.z * M_outer * K_outer + a_row * K_outer + a_col];

    int b_row = kk + threadIdx.y;
    int b_col = n_base + threadIdx.x;
    float bkn = B[blockIdx.z * K_outer * N_outer + b_row * N_outer + b_col];

    dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
  }
  for (int i = 0; i < 4; ++i) {
    int local_row = threadIdx.y * 4 + i;
    int local_col = threadIdx.x;

    int global_row = m_base + local_row;
    int global_col = n_base + local_col;
    if (global_row < M_outer && global_col < N_outer) {
      D[blockIdx.z * M_outer * N_outer + global_row * N_outer + global_col] =
          dmn[i];
    }
  }
}

extern "C" void bmm_kernel(float *A, float *B, float *C, int b, int m, int k,
                           int n) {
  float *d_A;
  float *d_B;
  float *d_C;

  hipMalloc(&d_A, b * m * k * sizeof(float));
  hipMalloc(&d_B, b * k * n * sizeof(float));
  hipMalloc(&d_C, b * m * n * sizeof(float));

  hipMemcpy(d_A, A, b * m * k * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, b * k * n * sizeof(float), hipMemcpyHostToDevice);

  dim3 grid((m + M - 1) / M, (n + N - 1) / N, b);
  dim3 block(16, 4, 1);
  bmm<<<grid, block>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, b * m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
