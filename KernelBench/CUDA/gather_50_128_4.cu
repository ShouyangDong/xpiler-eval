// ============================================================ //
// 实例 2: (1000, 2048), indices=8
// ============================================================ //
void static_gather_1000_2048_8_cuda(const float* d_params,
                                    const int* d_indices,
                                    float* d_output) {
  constexpr int PARAMS_BATCH = 1000;
  constexpr int PARAMS_LEN   = 2048;
  constexpr int INDICES_LEN  = 8;

  constexpr int block_size = 8;
  constexpr int grid_size  = 1; // 8 线程即可

  static_gather_kernel<PARAMS_BATCH, PARAMS_LEN, INDICES_LEN>
    <<<grid_size, block_size>>>(d_params, d_indices, d_output);
}
