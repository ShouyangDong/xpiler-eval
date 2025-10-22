#define SEQ_LEN 2048
#define NUM_HEADS 16
#define HEAD_DIM 256
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// 使用 Tensor Core 计算 QK^T
__global__ void mha(const half *__restrict__ Q,
                    const half *__restrict__ K,
                    const half *__restrict__ V,
                    half *__restrict__ O) {
  // (batch, head)
  int b = blockIdx.z;
  int h = blockIdx.y;

  // tile 行索引
  int row_tile = blockIdx.x;

  // --------------------------
  // Shared memory for attention
  // --------------------------
  __shared__ half attn_tile[BLOCK_M * SEQ_LEN];
  __shared__ half softmax_tile[BLOCK_M * SEQ_LEN];

  // ========== 1️⃣ 计算 QK^T ==========
  for (int row = row_tile * BLOCK_M; row < (row_tile + 1) * BLOCK_M; row += BLOCK_M) {
    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> s_frag;

    wmma::fill_fragment(s_frag, 0.0f);

    const half *q_ptr = Q + (((b * NUM_HEADS + h) * SEQ_LEN + row) * HEAD_DIM);
    const half *k_ptr = K + (((b * NUM_HEADS + h) * SEQ_LEN) * HEAD_DIM);

    for (int k = 0; k < HEAD_DIM; k += BLOCK_K) {
      wmma::load_matrix_sync(q_frag, q_ptr + k, HEAD_DIM);
      wmma::load_matrix_sync(k_frag, k_ptr + k * SEQ_LEN, SEQ_LEN);
      wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
    }

    wmma::store_matrix_sync(attn_tile + row * SEQ_LEN, s_frag, SEQ_LEN, wmma::mem_row_major);
  }

  // ========== 2️⃣ softmax ==========
  half scale = 1.0f / sqrtf((half)HEAD_DIM);
  for (int i = 0; i < BLOCK_M; ++i) {
    half max_val = -1e9f;
    for (int j = 0; j < SEQ_LEN; ++j)
      max_val = fmaxf(max_val, attn_tile[i * SEQ_LEN + j]);

    float sum_val = 0.0f;
    for (int j = 0; j < SEQ_LEN; ++j) {
      float e = expf(__half2float(attn_tile[i * SEQ_LEN + j]) * scale - max_val);
      softmax_tile[i * SEQ_LEN + j] = __float2half(e);
      sum_val += e;
    }
    for (int j = 0; j < SEQ_LEN; ++j)
      softmax_tile[i * SEQ_LEN + j] =
          __float2half(__half2float(softmax_tile[i * SEQ_LEN + j]) / sum_val);
  }

  // ========== 3️⃣ O = softmax(QK^T) @ V ==========
  for (int row = row_tile * BLOCK_M; row < (row_tile + 1) * BLOCK_M; row += BLOCK_M) {
    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> s_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::col_major> v_frag;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> o_frag;

    wmma::fill_fragment(o_frag, 0.0f);

    const half *s_ptr = softmax_tile + row * SEQ_LEN;
    const half *v_ptr = V + (((b * NUM_HEADS + h) * SEQ_LEN) * HEAD_DIM);

    for (int k = 0; k < SEQ_LEN; k += BLOCK_K) {
      wmma::load_matrix_sync(s_frag, s_ptr + k, SEQ_LEN);
      wmma::load_matrix_sync(v_frag, v_ptr + k * HEAD_DIM, HEAD_DIM);
      wmma::mma_sync(o_frag, s_frag, v_frag, o_frag);
    }

    wmma::store_matrix_sync(O + (((b * NUM_HEADS + h) * SEQ_LEN + row) * HEAD_DIM),
                            o_frag, HEAD_DIM, wmma::mem_row_major);
  }
}

extern "C" void mha_kernel_fp16_tensorcore(
    half *h_Q, half *h_K, half *h_V, half *h_output,
    int batch_size, int seq_len, int num_heads, int head_dim) {

  size_t size = (size_t)batch_size * seq_len * num_heads * head_dim;
  half *d_Q, *d_K, *d_V, *d_output;
  cudaMalloc(&d_Q, size * sizeof(half));
  cudaMalloc(&d_K, size * sizeof(half));
  cudaMalloc(&d_V, size * sizeof(half));
  cudaMalloc(&d_output, size * sizeof(half));

  cudaMemcpy(d_Q, h_Q, size * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K, size * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, size * sizeof(half), cudaMemcpyHostToDevice);

  dim3 grid(seq_len / TILE_M, num_heads, batch_size);
  dim3 block(32, 1, 1);  // 每 warp 一个 tile

  mha<<<grid, block>>>(d_Q, d_K, d_V, d_output);

  cudaMemcpy(h_output, d_output, size * sizeof(half), cudaMemcpyDeviceToHost);

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_output);
}
