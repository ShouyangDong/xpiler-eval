/ ============================================================ //
// 实例 6: (80, 256), indices=10
// ============================================================ //
void static_gather_80_256_10_cuda(const float* d_params,
                                  const int* d_indices,
                                  float* d_output) {
  constexpr int PARAMS_BATCH = 80;
  constexpr int PARAMS_LEN   = 256;
  constexpr int INDICES_LEN  = 10;

  constexpr int block_size = 16;
  constexpr int grid_size  = (10 + 15) / 16; // 1 block

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}
