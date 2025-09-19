// ============================================================== //
// 实例 2: 小 batch 分类头 (num_classes=1000, feature=2048, batch=8)
// 如 ImageNet 推理
// ============================================================== //
extern "C" void gather(const float* params,
                               const int* indices,
                               float* output) {
  constexpr int PARAMS_BATCH = 1000;
  constexpr int PARAMS_LEN   = 2048;
  constexpr int INDICES_LEN  = 8;

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
