
float sigmoidf(float input) { return 1 / (1 + exp(-1 * input)); }
extern "C" void sigmoid(float *input, float *output) {
  for (int i = 0; i < 192; i++) {
    output[i] = sigmoidf(input[i]);
  }
}