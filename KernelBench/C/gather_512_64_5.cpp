// ============================================================== //
// 实例 3: 序列采样 - Transformer 解码时取 top-k (seq=512, dim=64, k=5)
// ============================================================== //
extern "C" void gather(const float* params,
                            const int* indices,
                            float* output) {
  constexpr int PARAMS_BATCH = 512;
  constexpr int PARAMS_LEN   = 64;
  constexpr int INDICES_LEN  = 5;

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