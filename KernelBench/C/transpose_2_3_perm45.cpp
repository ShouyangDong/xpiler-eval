extern "C" void transpose(float *input, float *output) {
  int s0_in = 2;
  int s1_in = 3;

  int s0_out = s4_in;  // dim 0 of output = dim 4 of input
  int s1_out = s5_in;  // dim 1 of output = dim 5 of input

  for (int o0 = 0; o0 < s0_out; o0++) {
  for (int o1 = 0; o1 < s1_out; o1++) {
    output[o0 * s1_out + o1] = input[o0 * 1 + o1 * 1];
  }
  }
}
