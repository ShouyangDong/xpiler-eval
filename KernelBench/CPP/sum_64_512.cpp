extern "C" void sum(float *input, float *output) {
  int rows = 64;
  int cols = 512;

  for (int i = 0; i < rows; i++) {
    output[i] = 0.0f;
    for (int j = 0; j < cols; j++) {
      output[i] += input[i * cols + j];
    }
  }
}
