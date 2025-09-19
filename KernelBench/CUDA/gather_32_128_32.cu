__global__ void static_gather_kernel(const float* params,
                                     const int* indices,
                                     float* output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= INDICES_LEN) return;

  int idx = indices[i];
  float* out_row = &output[i * PARAMS_LEN];

  if (idx < 0 || idx >= PARAMS_BATCH) {
    // 越界：置零
    for (int j = 0; j < PARAMS_LEN; ++j) {
      out_row[j] = 0.0f;
    }
  } else {
    const float* src_row = &params[idx * PARAMS_LEN];
    for (int j = 0; j < PARAMS_LEN; ++j) {
      out_row[j] = src_row[j];
    }
  }
}

// ============================================================ //
// 实例 1: (10000, 128), indices=32
// ============================================================ //
void static_gather_10000_128_32_cuda(const float* d_params,
                                     const int* d_indices,
                                     float* d_output) {
  constexpr int PARAMS_BATCH = 10000;
  constexpr int PARAMS_LEN   = 128;
  constexpr int INDICES_LEN  = 32;

  constexpr int block_size = 32;
  constexpr int grid_size  = (INDICES_LEN + block_size - 1) / block_size;

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}
