
extern "C" void min(float *input, float *output) {
  int s0 = 8;
  int s1 = 16;
  int s2 = 8;

  for (int i0 = 0; i0 < s0; i0++) {
    for (int i1 = 0; i1 < s1; i1++) {
      output[i0 * s1 + i1 * s2] = INFINITY;
    }
  }

  for (int i0 = 0; i0 < s0; i0++) {
    for (int i1 = 0; i1 < s1; i1++) {
      for (int i2 = 0; i2 < s2; i2++) {
        if (input[i0 * s1 + i1 * s2 + i2] < output[i0 * s1 + i1 * s2]) {
          output[i0 * s1 + i1 * s2] = input[i0 * s1 + i1 * s2 + i2];
        }
      }
    }
  }
}
