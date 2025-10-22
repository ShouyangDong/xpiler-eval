__global__ void mha_tensorcore_single_kernel(const half *__restrict__ Q,
                                             const half *__restrict__ K,
                                             const half *__restrict__ V,
                                             half *__restrict__ output,
                                             int seq_len, int head_dim) {
  int batch_head = blockIdx.z;
  int tile_m = blockIdx.y;
  int row_start = tile_m * WMMA_M;
  if (row_start >= seq_len) {
    return;
  }

  size_t head_stride = static_cast<size_t>(seq_len) * head_dim;
  const half *Q_base = Q + static_cast<size_t>(batch_head) * head_stride;
  const half *K_base = K + static_cast<size_t>(batch_head) * head_stride;
  const half *V_base = V + static_cast<size_t>(batch_head) * head_stride;
  half *O_base = output + static_cast<size_t>(batch_head) * head_stride;

  extern __shared__ unsigned char smem_raw[];
  float *score_tile = reinterpret_cast<float *>(smem_raw);
  half *prob_tile = reinterpret_cast<half *>(score_tile + WMMA_M * seq_len);
  float *out_tile = reinterpret_cast<float *>(prob_tile + WMMA_M * seq_len);

  for (int n_tile = 0; n_tile < seq_len; n_tile += WMMA_N) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = 0; k_tile < head_dim; k_tile += WMMA_K) {
      const half *a_ptr = Q_base + row_start * head_dim + k_tile;
      const half *b_ptr = K_base + n_tile * head_dim + k_tile;

      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                             half, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                             half, nvcuda::wmma::col_major>
          b_frag;

      nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, head_dim);
      nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, head_dim);
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    nvcuda::wmma::store_matrix_sync(score_tile + n_tile, c_frag, seq_len,
                                    nvcuda::wmma::mem_row_major);
  }

  __syncthreads();

  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  for (int idx = threadIdx.x; idx < WMMA_M * seq_len; idx += blockDim.x) {
    score_tile[idx] *= scale;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    for (int row = 0; row < WMMA_M; ++row) {
      float *row_ptr = score_tile + row * seq_len;
      float max_val = row_ptr[0];
      for (int col = 1; col < seq_len; ++col) {
        max_val = fmaxf(max_val, row_ptr[col]);
      }
      float sum = 0.0f;
      for (int col = 0; col < seq_len; ++col) {
        float val = expf(row_ptr[col] - max_val);
        row_ptr[col] = val;
        sum += val;
      }
      float inv_sum = 1.0f / sum;
      for (int col = 0; col < seq_len; ++col) {
        float prob = row_ptr[col] * inv_sum;
        row_ptr[col] = prob;
        prob_tile[row * seq_len + col] = __float2half(prob);
      }
    }
  }

  __syncthreads();

  for (int idx = threadIdx.x; idx < WMMA_M * head_dim; idx += blockDim.x) {
    out_tile[idx] = 0.0f;
  }

  __syncthreads();

  for (int n_tile = 0; n_tile < head_dim; n_tile += WMMA_N) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = 0; k_tile < seq_len; k_tile += WMMA_K) {
      const half *a_ptr = prob_tile + k_tile;
      const half *b_ptr = V_base + k_tile * head_dim + n_tile;

      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                             half, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                             half, nvcuda::wmma::row_major>
          b_frag;

      nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, seq_len);
      nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, head_dim);
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    nvcuda::wmma::store_matrix_sync(out_tile + n_tile, c_frag, head_dim,
                                    nvcuda::wmma::mem_row_major);
  }

  __syncthreads();

  for (int idx = threadIdx.x; idx < WMMA_M * head_dim; idx += blockDim.x) {
    int row = idx / head_dim;
    int col = idx % head_dim;
    if (row_start + row < seq_len) {
      O_base[(row_start + row) * head_dim + col] =
          __float2half(out_tile[idx]);
    }
  }
}


extern "C" void mha_kernel_fp16_tensorcore_single_kernel(
    const half *h_Q, const half *h_K, const half *h_V, half *h_output,
    int batch_size, int seq_len, int num_heads, int head_dim) {
  if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
    throw std::invalid_argument("All dimensions must be positive");
  }

  if ((seq_len % WMMA_M) != 0 || (head_dim % WMMA_K) != 0) {
    throw std::invalid_argument(
        "seq_len and head_dim must be multiples of 16 for Tensor Cores");
  }

  int batch_count = batch_size * num_heads;
  size_t matrix_elems = static_cast<size_t>(batch_count) * seq_len * head_dim;

  half *d_Q = nullptr;
  half *d_K = nullptr;
  half *d_V = nullptr;
  half *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_Q, matrix_elems * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_K, matrix_elems * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_V, matrix_elems * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_output, matrix_elems * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(d_Q, h_Q, matrix_elems * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K, matrix_elems * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V, matrix_elems * sizeof(half),
                        cudaMemcpyHostToDevice));

  dim3 block(32, 1, 1);
  dim3 grid(1, seq_len / WMMA_M, batch_count);

  size_t shared_bytes = static_cast<size_t>(WMMA_M) * seq_len *
                            (sizeof(float) + sizeof(half)) +
                        static_cast<size_t>(WMMA_M) * head_dim * sizeof(float);

  mha_tensorcore_single_kernel<<<grid, block, shared_bytes>>>(
      d_Q, d_K, d_V, d_output, seq_len, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_output, d_output, matrix_elems * sizeof(half),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_output);
}