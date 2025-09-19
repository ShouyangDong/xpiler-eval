
extern "C" void min(float *input, float *output) {
  int s0 = 64;

  output[0] = INFINITY;

  // Compare along last dimension (dim=-1)
  for (int i0 = 0; i0 < s0; i0++) {
      if (input[0] < output[0]) {
        output[0] = input[0];
      }
  }
}
