
float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

extern "C" void gelu(float *input, float *output) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 5; k++) {
        int index = i * 4 * 5 + j * 5 + k;
        output[index] = geluf(input[index]);
      }
    }
  }
}