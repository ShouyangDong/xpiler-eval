extern "C" void transpose(float *input, float *output) {
  int s0_in = 1;

  int s0_out = s0_in;  // dim 0 of output = dim 0 of input

  for (int o0 = 0; o0 < s0_out; o0++) {
    output[o0] = input[o0 * 1];
  }
}
