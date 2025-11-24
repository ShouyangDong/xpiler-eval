#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

#define BATCH 2
#define HEAD 4
#define SEQ_Q 16
#define SEQ_KV 512
#define HEAD_DIM 16

__global__ void gqa(const half *__restrict__ Q, const half *__restrict__ K,
                    const half *__restrict__ V, half *__restrict__ O) {
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int tile_row = blockIdx.x;

  __shared__ half attn_tile[BLOCK_M * SEQ_KV];
  __shared__ half softmax_tile[BLOCK_M * SEQ_KV];

  for (int row = 0; row < SEQ_Q; row += BLOCK_M) {
    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half,
                   wmma::row_major>
        q_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half,
                   wmma::col_major>
        k_frag;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> s_frag;

    wmma::fill_fragment(s_frag, 0.0f);

    const half *q_ptr = Q + (batch)*SEQ_Q * HEAD_DIM + row * HEAD_DIM;
    const half *k_ptr = K + (batch)*SEQ_KV * HEAD_DIM;

    for (int k = 0; k < HEAD_DIM; k += BLOCK_K) {
      wmma::load_matrix_sync(q_frag, q_ptr + k, HEAD_DIM);
      wmma::load_matrix_sync(k_frag, k_ptr + k * SEQ_KV, SEQ_KV);
      wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
    }

    wmma::store_matrix_sync(attn_tile + row * SEQ_KV, s_frag, SEQ_KV,
                            wmma::mem_row_major);
  }

  for (int i = 0; i < SEQ_Q; ++i) {
    half max_val = -1e9f;
    for (int j = 0; j < SEQ_KV; ++j)
      max_val = fmaxf(max_val, attn_tile[i * SEQ_KV + j]);

    half sum_val = 0.0f;
    for (int j = 0; j < SEQ_KV; ++j) {
      half e = expf(attn_tile[i * SEQ_KV + j] - max_val);
      softmax_tile[i * SEQ_KV + j] = e;
      sum_val += e;
    }
    for (int j = 0; j < SEQ_KV; ++j)
      softmax_tile[i * SEQ_KV + j] /= sum_val;
  }

  for (int row = 0; row < SEQ_Q; row += BLOCK_M) {
    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half,
                   wmma::row_major>
        s_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half,
                   wmma::col_major>
        v_frag;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> o_frag;

    wmma::fill_fragment(o_frag, 0.0f);

    const half *s_ptr = softmax_tile + row * SEQ_KV;
    const half *v_ptr = V + (batch)*SEQ_KV * HEAD_DIM;

    for (int k = 0; k < SEQ_KV; k += BLOCK_K) {
      wmma::load_matrix_sync(s_frag, s_ptr + k, SEQ_KV);
      wmma::load_matrix_sync(v_frag, v_ptr + k * HEAD_DIM, HEAD_DIM);
      wmma::mma_sync(o_frag, s_frag, v_frag, o_frag);
    }

    wmma::store_matrix_sync(O + (batch)*SEQ_Q * HEAD_DIM + row * HEAD_DIM,
                            o_frag, HEAD_DIM, wmma::mem_row_major);
  }
}

extern "C" void gqa_kernel(half *Q, half *K, half *V, half *O, int batch,
                           int heads, int M, int K_dim, int N) {
  half *d_Q, *d_K, *d_V, *d_O;
  size_t q_sz = batch * heads * M * K_dim * sizeof(half);
  size_t k_sz = batch * heads * K_dim * N * sizeof(half);
  size_t v_sz = batch * heads * N * K_dim * sizeof(half);
  size_t o_sz = batch * heads * M * K_dim * sizeof(half);

  cudaMalloc(&d_Q, q_sz);
  cudaMalloc(&d_K, k_sz);
  cudaMalloc(&d_V, v_sz);
  cudaMalloc(&d_O, o_sz);

  cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, K, k_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, v_sz, cudaMemcpyHostToDevice);

  dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_K - 1) / BLOCK_K,
            batch * heads);
  dim3 block(32);
  gqa<<<grid, block>>>(d_Q, d_K, d_V, d_O);
  cudaMemcpy(O, d_O, o_sz, cudaMemcpyDeviceToHost);

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);
}
