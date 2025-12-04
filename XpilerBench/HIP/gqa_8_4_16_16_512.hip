#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 4

#define M_outer 16
#define K_outer 16
#define N_outer 512

__global__ void gqa(float *Q, float *K, float *V, float *O) {
  int bid = blockIdx.z; // batch * heads
  int m_base = blockIdx.x * BLOCK_M;
  int n_base = blockIdx.y * BLOCK_N;
  using float4 =
      __attribute__((__vector_size__(BLOCK_K * sizeof(float)))) float;
  float4 dmn = {0};
  __shared__ float temp[M_outer * N_outer];
  __shared__ float sum_array[N_outer];
  for (int kk = 0; kk < K_outer; kk += BLOCK_K) {
    int a_row = m_base + threadIdx.x;
    int a_col = kk + threadIdx.y;
    int b_row = kk + threadIdx.y;
    int b_col = n_base + threadIdx.x;
    if (a_row < M_outer && a_col < K_outer && b_row < K_outer &&
        b_col < N_outer) {
      float amk = Q[bid * M_outer * K_outer + a_row * K_outer + a_col];
      float bkn = K[bid * K_outer * N_outer + b_row * N_outer + b_col];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
  }

  for (int i = 0; i < 4; ++i) {
    int local_row = threadIdx.y * 4 + i;
    int local_col = threadIdx.x;

    int global_row = m_base + local_row;
    int global_col = n_base + local_col;
    if (global_row < M_outer && global_col < N_outer) {
      temp[global_row * N_outer + global_col] = dmn[i];
    }

    // The Softmax code:
    float sum = 0.0f;
    for (int i_ex = 0; i_ex < N_outer; ++i_ex) {
      sum_array[i_ex] = expf(temp[global_row * N_outer + i_ex]);
    }
    for (int i_sf = 0; i_sf < N_outer; ++i_sf) {
      sum += sum_array[i_sf];
    }
    for (int k_sf = 0; k_sf < N_outer; ++k_sf) {
      temp[global_row * N_outer + k_sf] =
          sum_array[global_row * N_outer + k_sf] / sum;
    }
  }
  dmn = {0};
  for (int kk = 0; kk < K_outer; kk += BLOCK_K) {
    int a_row = m_base + threadIdx.x;
    int a_col = kk + threadIdx.y;
    int c_row = kk + threadIdx.y;
    int c_col = n_base + threadIdx.x;
    if (a_row < M_outer && a_col < N_outer && c_row < N_outer &&
        c_col < K_outer) {
      float amk = temp[bid * M_outer * K_outer + a_row * K_outer + a_col];
      float bkn = V[bid * K_outer * N_outer + c_row * N_outer + c_col];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
  }

  for (int i = 0; i < 4; ++i) {
    int local_row = threadIdx.y * 4 + i;
    int local_col = threadIdx.x;

    int global_row = m_base + local_row;
    int global_col = n_base + local_col;
    if (global_row < M_outer && global_col < K_outer) {
      O[global_row * N_outer + global_col] = dmn[i];
    }
  }
}

extern "C" void gqa_kernel(float *Q, float *K, float *V, float *O, int batch,
                           int heads, int M, int N, int K_dim) {
  float *d_Q, *d_K, *d_V, *d_O;
  size_t q_sz = batch * heads * M * K_dim * sizeof(float);
  size_t k_sz = batch * heads * K_dim * N * sizeof(float);
  size_t v_sz = batch * heads * N * K_dim * sizeof(float);
  size_t o_sz = batch * heads * M * K_dim * sizeof(float);

  hipMalloc(&d_Q, q_sz);
  hipMalloc(&d_K, k_sz);
  hipMalloc(&d_V, v_sz);
  hipMalloc(&d_O, o_sz);

  hipMemcpy(d_Q, Q, q_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_K, K, k_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_V, V, v_sz, hipMemcpyHostToDevice);
  hipMemset(d_O, 0, o_sz); // zero out

  // Grid: M_blocks x N_blocks x (batch * heads)
  dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_K - 1) / BLOCK_K,
            batch * heads);
  dim3 block(16, 4, 1);

  gqa<<<grid, block>>>(d_Q, d_K, d_V, d_O);

  hipMemcpy(O, d_O, o_sz, hipMemcpyDeviceToHost);

  hipFree(d_Q);
  hipFree(d_K);
  hipFree(d_V);
  hipFree(d_O);
}
