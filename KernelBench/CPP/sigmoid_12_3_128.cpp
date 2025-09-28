
float sigmoidf(float input) { return 1 / (1 + exp(-1 * input)); }
extern "C" void sigmoid(float *input, float *output) {
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 128; k++) {
        output[i * 3 * 128 + j * 128 + k] =
            sigmoidf(input[i * 3 * 128 + j * 128 + k]);
      }
    }
  }
}