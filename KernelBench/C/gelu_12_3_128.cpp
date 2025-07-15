
float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
extern "C" void gelu(float *input, float *output) {
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 128; k++) {
        int index = i * 3 * 128 + j * 128 + k;
        output[index] = geluf(input[index]);
      }
    }
  }
}