extern "C" void min(float *input, float *output) {
  const int s0 = 4;
  const int s1 = 64;
  const int s2 = 64;

  for (int i0 = 0; i0 < s0; ++i0) {
    for (int i2 = 0; i2 < s2; ++i2) {
      output[i0 * s2 + i2] = INFINITY;
    }
  }

  for (int i0 = 0; i0 < s0; ++i0) {
    for (int i1 = 0; i1 < s1; ++i1) {
      for (int i2 = 0; i2 < s2; ++i2) {
        float val = input[i0 * s1 * s2 + i1 * s2 + i2];
        if (val < output[i0 * s2 + i2]) {
          output[i0 * s2 + i2] = val;
        }
      }
    }
  }
}