
float sigmoidf(float input) { return 1 / (1 + exp(-1 * input)); }
extern "C" void sigmoid(float *input, float *output) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 5; k++) {
        output[i * 4 * 5 + j * 5 + k] = sigmoidf(input[i * 4 * 5 + j * 5 + k]);
      }
    }
  }
}