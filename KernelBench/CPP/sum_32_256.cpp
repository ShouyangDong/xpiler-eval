extern "C" void sum(float *input, float *output) {
  const int rows = 32;
  const int cols = 256;

  for (int j = 0; j < cols; j++) {
    output[j] = 0.0f;
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[j] += input[i * cols + j];
    }
  }
}
