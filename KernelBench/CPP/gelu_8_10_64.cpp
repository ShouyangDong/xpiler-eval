
float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
extern "C" void gelu(float *input, float *output) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 64; k++) {
        int index = i * 10 * 64 + j * 64 + k;
        output[index] = geluf(input[index]);
      }
    }
  }
}