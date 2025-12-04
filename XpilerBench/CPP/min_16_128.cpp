extern "C" void min(float *input, float *output) {
  int s0 = 16;
  int s1 = 128;

  for (int i0 = 0; i0 < s0; i0++) {
    output[i0] = INFINITY;
  }

  for (int i0 = 0; i0 < s0; i0++) {
    for (int i1 = 0; i1 < s1; i1++) {
      float val = input[i0 * s1 + i1];
      if (val < output[i0]) {
        output[i0] = val;
      }
    }
  }
}
