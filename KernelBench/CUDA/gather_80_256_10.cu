// ============================================================ //
// 实例 3: (512, 64), indices=5
// ============================================================ //
void static_gather_512_64_5_cuda(const float* d_params,
                                 const int* d_indices,
                                 float* d_output) {
  constexpr int PARAMS_BATCH = 512;
  constexpr int PARAMS_LEN   = 64;
  constexpr int INDICES_LEN  = 5;

  constexpr int block_size = 8;
  constexpr int grid_size  = 1;

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}