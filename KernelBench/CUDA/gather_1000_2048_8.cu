// ============================================================ //
// 实例 7: (128, 512), indices=3
// ============================================================ //
void static_gather_128_512_3_cuda(const float* d_params,
                                  const int* d_indices,
                                  float* d_output) {
  constexpr int PARAMS_BATCH = 128;
  constexpr int PARAMS_LEN   = 512;
  constexpr int INDICES_LEN  = 3;

  constexpr int block_size = 8;
  constexpr int grid_size  = 1;

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}