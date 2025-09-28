extern "C" void transpose(float *input, float *output) {
  int s0_in = 36;
  int s1_in = 16;
  int s2_in = 48;

  int s0_out = s2_in;
  int s1_out = s0_in;

  for (int o0 = 0; o0 < s0_out; ++o0) {
    for (int o1 = 0; o1 < s1_out; ++o1) {

      output[o0 * s1_out + o1] = input[o1 * (s1_in * s2_in) + 0 * s2_in + o0];
    }
  }
}
