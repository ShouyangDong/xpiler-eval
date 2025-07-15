
float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
extern "C" void gelu(float *input, float *output) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 128; j++) {
      int index = i * 128 + j;
      output[index] = geluf(input[index]);
    }
  }
}