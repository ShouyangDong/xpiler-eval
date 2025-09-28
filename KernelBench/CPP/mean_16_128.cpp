extern "C" void mean(float *input, float *output) {
  int rows = 16;
  int cols = 128;

  for (int i = 0; i < 1 * 128; i++) {
    output[i] = 0.0f;
  }

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      output[j] += input[i * cols + j];
    }
    output[j] /= 16;
  }
}
