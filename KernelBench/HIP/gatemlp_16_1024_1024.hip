#define M 16
#define N 16
#define K 4

#define M_outer 16
#define N_outer 1024
#define K_outer 1024

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void gatemlp(const float *A, const float *B, const float *C,
                      float *D) {
  int m_base = blockIdx.x * M;
  int n_base = blockIdx.y * N;
  using float4 = __attribute__((__vector_size__(K * sizeof(float)))) float;
  float4 dmn_AB = {0};
  float4 dmn_AC = {0};
  for (int kk = 0; kk < K_outer; kk += K) {
    int a_row = m_base + threadIdx.x;
    int a_col = kk + threadIdx.y;
    float amk = A[a_row * K_outer + a_col];

    int b_row = kk + threadIdx.y;
    int b_col = n_base + threadIdx.x;
    float bkn = B[b_row * N_outer + b_col];
    float ckn = C[b_row * N_outer + b_col];
    dmn_AB = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn_AB, 0, 0, 0);
    dmn_AC = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, ckn, dmn_AC, 0, 0, 0);
  }

  float4 result;
  for (int i = 0; i < 4; ++i) {
      float gate = silu(dmn_AB[i]);
      result[i] = gate * dmn_AC[i];
  }

  for (int i = 0; i < 4; ++i) {
    int local_row = threadIdx.y * 4 + i;
    int local_col = threadIdx.x;

    int global_row = m_base + local_row;
    int global_col = n_base + local_col;
    if (global_row < M_outer && global_col < N_outer) {
      D[global_row * N_outer + global_col] = result[i];
    }
  }
}

extern "C" void gatemlp_kernel(float *A, float *B, float *C, float *D, int m,
                             int k, int n) {
  float *d_A;
  float *d_B;
  float *d_C;
  float *d_D;

  hipMalloc(&d_A, m * k * sizeof(float));
  hipMalloc(&d_B, k * n * sizeof(float));
  hipMalloc(&d_C, k * n * sizeof(float));
  hipMalloc(&d_D, m * n * sizeof(float));

  hipMemcpy(d_A, A, m * k * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, k * n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_C, C, k * n * sizeof(float), hipMemcpyHostToDevice);
  dim3 grid((m + M - 1) / M, (n + N - 1) / N);
  dim3 block(16, 4, 1);
  gatemlp<<<grid, block>>>(d_A, d_B, d_C, d_D);

  hipMemcpy(D, d_D, m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
  hipFree(d_D);
}
