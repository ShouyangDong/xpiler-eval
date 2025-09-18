// ============================================================ //
// 实例 4: (100, 32), indices=16
// ============================================================ //
void static_gather_100_32_16_cuda(const float* d_params,
                                  const int* d_indices,
                                  float* d_output) {
  constexpr int PARAMS_BATCH = 100;
  constexpr int PARAMS_LEN   = 32;
  constexpr int INDICES_LEN  = 16;

  constexpr int block_size = 16;
  constexpr int grid_size  = 1;

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}