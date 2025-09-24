
extern "C" void min(float *input, float *output) {
  int s0 = 16;
  int s1 = 128;

  for (int i0 = 0; i0 < s0; i0++) {
    output[i0 * s1] = INFINITY;
  }

  for (int i0 = 0; i0 < s0; i0++) {
    for (int i1 = 0; i1 < s1; i1++) {
      if (input[i0 * s1 + i1] < output[i0 * s1]) {
        output[i0 * s1] = input[i0 * s1 + i1];
      }
    }
  }
}
