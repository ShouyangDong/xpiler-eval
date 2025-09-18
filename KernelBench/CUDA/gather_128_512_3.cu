// ============================================================ //
// 实例 5: (50, 128), indices=4
// ============================================================ //
void static_gather_50_128_4_cuda(const float* d_params,
                                 const int* d_indices,
                                 float* d_output) {
  constexpr int PARAMS_BATCH = 50;
  constexpr int PARAMS_LEN   = 128;
  constexpr int INDICES_LEN  = 4;

  constexpr int block_size = 8;
  constexpr int grid_size  = 1;

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}
