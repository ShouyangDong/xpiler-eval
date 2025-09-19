// ============================================================== //
// 实例 6: 多实例预测 - 取 top-10 预测 (classes=80, features=256, topk=10)
// 如目标检测后处理
// ============================================================== //
extern "C" void static_gather_80_256_10(const float* params,
                             const int* indices,
                             float* output) {
  constexpr int PARAMS_BATCH = 80;
  constexpr int PARAMS_LEN   = 256;
  constexpr int INDICES_LEN  = 10;

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