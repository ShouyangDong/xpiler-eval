extern "C" void transpose(float *input, float *output) {
  int s0_in = 24;
  int s1_in = 36;

  int s0_out = 36;
  int s1_out = 24;

  for (int o0 = 0; o0 < s0_out; o0++) {
    for (int o1 = 0; o1 < s1_out; o1++) {

      output[o0 * s1_out + o1] = input[o1 * s1_in + o0];
    }
  }
}