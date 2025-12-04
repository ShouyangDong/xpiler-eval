extern "C" void min(float *input, float *output) {
  const int s0 = 2;
  const int s1 = 4;
  const int s2 = 5;
  const int s3 = 64;

  for (int i0 = 0; i0 < s0; ++i0) {
    for (int i2 = 0; i2 < s2; ++i2) {
      for (int i3 = 0; i3 < s3; ++i3) {
        output[i0 * s2 * s3 + i2 * s3 + i3] = INFINITY;
      }
    }
  }

  for (int i0 = 0; i0 < s0; ++i0) {
    for (int i1 = 0; i1 < s1; ++i1) {
      for (int i2 = 0; i2 < s2; ++i2) {
        for (int i3 = 0; i3 < s3; ++i3) {
          float val = input[i0 * s1 * s2 * s3 + i1 * s2 * s3 + i2 * s3 + i3];
          int out_idx = i0 * s2 * s3 + i2 * s3 + i3;
          if (val < output[out_idx]) {
            output[out_idx] = val;
          }
        }
      }
    }
  }
}
