

extern "C" void gather(const float *params, const int *indices, float *output) {
  constexpr int PARAMS_BATCH = 100;
  constexpr int PARAMS_LEN = 32;
  constexpr int INDICES_LEN = 16;

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