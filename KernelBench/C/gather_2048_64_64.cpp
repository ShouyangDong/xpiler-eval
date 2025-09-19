// ============================================================== //
// 实例 8: 批量索引 - 多样本特征提取 (samples=2048, feature_dim=64, batch=64)
// 如数据采样系统
// ============================================================== //
extern "C" void gather(const float* params,
                              const int* indices,
                              float* output) {
  constexpr int PARAMS_BATCH = 2048;
  constexpr int PARAMS_LEN   = 64;
  constexpr int INDICES_LEN  = 64;

  for (int i = 0; i < INDICES_LEN; ++i) {
    int idx = indices[i];
    if (idx < 0 || idx >= PARAMS_BATCH) {
      for (int j = 0; j < PARAMS_LEN; ++j) {
        output[i * PARAMS_LEN + j] = 0.0f;
      }
    } else {
      for (int j = 0; j < PARAMS_LEN; ++j) {
        output[i * PARAMS_LEN + j] = params[idx * PARAMS_LEN + j];
      }
    }
  }
}
