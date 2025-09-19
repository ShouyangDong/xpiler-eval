
extern "C" void min(float *input, float *output) {
  int s0 = 4;
  int s1 = 32;

  // Initialize output to INFINITY
  for (int i0 = 0; i0 < s0; i0++) {
    output[i0 * s1] = INFINITY;
  }

  // Compare along last dimension (dim=-1)
  for (int i0 = 0; i0 < s0; i0++) {
  for (int i1 = 0; i1 < s1; i1++) {
      if (input[i0 * s1 + i1] < output[i0 * s1]) {
        output[i0 * s1] = input[i0 * s1 + i1];
      }
  }
  }
}
