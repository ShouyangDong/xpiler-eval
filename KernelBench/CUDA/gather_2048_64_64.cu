// ============================================================ //
// 实例 8: (2048, 64), indices=64
// ============================================================ //
void static_gather_2048_64_64_cuda(const float* d_params,
                                   const int* d_indices,
                                   float* d_output) {
  constexpr int PARAMS_BATCH = 2048;
  constexpr int PARAMS_LEN   = 64;
  constexpr int INDICES_LEN  = 64;

  constexpr int block_size = 64;
  constexpr int grid_size  = 1;

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}
