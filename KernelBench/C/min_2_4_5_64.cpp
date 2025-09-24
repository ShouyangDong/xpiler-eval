
extern "C" void min(float *input, float *output) {
  int s0 = 2;
  int s1 = 4;
  int s2 = 5;
  int s3 = 6;

  for (int i0 = 0; i0 < s0; i0++) {
    for (int i1 = 0; i1 < s1; i1++) {
      for (int i2 = 0; i2 < s2; i2++) {
        output[i0 * s1 + i1 * s2 + i2 * s3] = INFINITY;
      }
    }
  }

  for (int i0 = 0; i0 < s0; i0++) {
    for (int i1 = 0; i1 < s1; i1++) {
      for (int i2 = 0; i2 < s2; i2++) {
        for (int i3 = 0; i3 < s3; i3++) {
          if (input[i0 * s1 + i1 * s2 + i2 * s3 + i3] <
              output[i0 * s1 + i1 * s2 + i2 * s3]) {
            output[i0 * s1 + i1 * s2 + i2 * s3] =
                input[i0 * s1 + i1 * s2 + i2 * s3 + i3];
          }
        }
      }
    }
  }
}
