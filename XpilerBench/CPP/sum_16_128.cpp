extern "C" void sum(float *input, float *output) {
  int rows = 16;
  int cols = 128;

  for (int i = 0; i < rows; i++) {
    output[i] = 0.0f;
    for (int j = 0; j < cols; j++) {
      output[i] += input[i * cols + j];
    }
  }
}
