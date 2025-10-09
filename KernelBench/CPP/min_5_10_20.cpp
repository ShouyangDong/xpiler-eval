extern "C" void min(float *input, float *output) {
  const int s0 = 5;
  const int s1 = 10;
  const int s2 = 20;

  for (int i0 = 0; i0 < s0; ++i0) {
    for (int i2 = 0; i2 < s2; ++i2) {
      output[i0 * s2 + i2] = INFINITY;
    }
  }

  for (int i0 = 0; i0 < s0; ++i0) {
    for (int i1 = 0; i1 < s1; ++i1) {
      size_t base = (size_t)i0 * s1 * s2 + (size_t)i1 * s2;
      size_t out_base = (size_t)i0 * s2;
      for (int i2 = 0; i2 < s2; ++i2) {
        float val = input[base + i2];
        if (val < output[out_base + i2]) {
          output[out_base + i2] = val;
        }
      }
    }
  }
}
