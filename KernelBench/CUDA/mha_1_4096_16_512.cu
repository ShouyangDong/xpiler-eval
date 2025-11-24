#define SEQ_LEN 16
#define NUM_HEADS 4096
#define HEAD_DIM 512
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

__global__ void mha_fixed(const half *__restrict__ Q,
                          const half *__restrict__ K,
                          const half *__restrict__ V, half *__restrict__ O) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int row_tile = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= 32)
    return;

  __shared__ half attn_tile[BLOCK_M * SEQ_LEN];
  __shared__ float softmax_row_max[BLOCK_M];
  __shared__ float softmax_row_sum[BLOCK_M];

  int row_start = row_tile * BLOCK_M;

  wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half,
                 wmma::row_major>
      q_frag;

  wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half,
                 wmma::col_major>
      k_frag;

  wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> s_frag;

  wmma::fill_fragment(s_frag, 0.0f);

  const half *q_base_ptr =
      Q + (((size_t)b * NUM_HEADS + h) * SEQ_LEN + row_start) * HEAD_DIM;

  const half *k_base_ptr =
      K + (((size_t)b * NUM_HEADS + h) * SEQ_LEN) * HEAD_DIM;

  for (int k = 0; k < HEAD_DIM; k += BLOCK_K) {

    wmma::load_matrix_sync(q_frag, q_base_ptr + k, HEAD_DIM);

    wmma::load_matrix_sync(k_frag, k_base_ptr + k, HEAD_DIM);

    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
  }

  if (tid < 32) {

    wmma::store_matrix_sync(attn_tile, s_frag, SEQ_LEN, wmma::mem_row_major);
  }
  __syncthreads();

  float scale = 1.0f / sqrtf((float)HEAD_DIM);

  for (int i = tid; i < BLOCK_M; i += 32) {
    float max_val = -1e9f;
    for (int j = 0; j < SEQ_LEN; ++j) {
      max_val =
          fmaxf(max_val, __half2float(attn_tile[i * SEQ_LEN + j]) * scale);
    }
    softmax_row_max[i] = max_val;
  }

  int r = tid / SEQ_LEN;
  int c = tid % SEQ_LEN;

  if (tid < BLOCK_M) {
    int i = tid;
    float max_val = -1e9f;
    float sum_val = 0.0f;

    for (int j = 0; j < SEQ_LEN; ++j) {
      max_val =
          fmaxf(max_val, __half2float(attn_tile[i * SEQ_LEN + j]) * scale);
    }
    softmax_row_max[i] = max_val;
  }
  __syncthreads();

  if (tid < BLOCK_M) {
    int i = tid;
    float max_val = softmax_row_max[i];
    float sum_val = 0.0f;

    for (int j = 0; j < SEQ_LEN; ++j) {
      float e =
          expf(__half2float(attn_tile[i * SEQ_LEN + j]) * scale - max_val);

      sum_val += e;
    }
    softmax_row_sum[i] = 1.0f / sum_val;
  }
  __syncthreads();

  for (int idx = tid; idx < BLOCK_M * SEQ_LEN; idx += 32) {
    int r_idx = idx / SEQ_LEN;
    float max_val = softmax_row_max[r_idx];
    float inv_sum = softmax_row_sum[r_idx];

    float e = expf(__half2float(attn_tile[idx]) * scale - max_val);
    attn_tile[idx] = __float2half(e * inv_sum);
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half,
                 wmma::row_major>
      s_frag;

  wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half,
                 wmma::col_major>
      v_frag;

  wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> o_frag;

  wmma::fill_fragment(o_frag, 0.0f);

  const half *v_base_ptr =
      V + (((size_t)b * NUM_HEADS + h) * SEQ_LEN) * HEAD_DIM;

  const half *s_ptr = attn_tile;

  for (int k = 0; k < HEAD_DIM; k += BLOCK_K) {

    for (int col_tile = 0; col_tile < HEAD_DIM; col_tile += BLOCK_N) {

      wmma::fill_fragment(o_frag, 0.0f);

      for (int k = 0; k < SEQ_LEN; k += BLOCK_K) {

        wmma::load_matrix_sync(s_frag, s_ptr + k, SEQ_LEN);

        wmma::load_matrix_sync(v_frag, v_base_ptr + k * HEAD_DIM + col_tile,
                               HEAD_DIM);

        wmma::mma_sync(o_frag, s_frag, v_frag, o_frag);
      }

      const half *o_base_ptr =
          O + (((size_t)b * NUM_HEADS + h) * SEQ_LEN + row_start) * HEAD_DIM;

      if (tid < 32) {
        wmma::store_matrix_sync(o_base_ptr + col_tile, o_frag, HEAD_DIM,
                                wmma::mem_row_major);
      }
    }
  }
}

extern "C" void mha_kernel(half *h_Q, half *h_K, half *h_V, half *h_output,
                           int batch_size, int num_heads, int seq_len,
                           int head_dim) {

  size_t size = (size_t)batch_size * seq_len * num_heads * head_dim;
  half *d_Q, *d_K, *d_V, *d_output;
  cudaMalloc(&d_Q, size * sizeof(half));
  cudaMalloc(&d_K, size * sizeof(half));
  cudaMalloc(&d_V, size * sizeof(half));
  cudaMalloc(&d_output, size * sizeof(half));

  cudaMemcpy(d_Q, h_Q, size * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K, size * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, size * sizeof(half), cudaMemcpyHostToDevice);

  dim3 grid(seq_len / BLOCK_M, num_heads, batch_size);
  dim3 block(32, 1, 1);

  mha<<<grid, block>>>(d_Q, d_K, d_V, d_output);

  cudaMemcpy(h_output, d_output, size * sizeof(half), cudaMemcpyDeviceToHost);

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_output);
}
