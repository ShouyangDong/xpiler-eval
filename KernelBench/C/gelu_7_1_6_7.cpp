
float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
extern "C" void gelu(float *input, float *output) {
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 6; k++) {
        for (int l = 0; l < 7; l++) {
          int index = i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l;
          output[index] = geluf(input[index]);
        }
      }
    }
  }
}