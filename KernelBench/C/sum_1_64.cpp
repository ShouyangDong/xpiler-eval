extern "C" void sum(float *input, float *output) {
  int rows = 1;
  int cols = 64;

  for (int i = 0; i < rows; i++) {
    output[i] = 0;
    for (int j = 0; j < cols; j++) {
      output[i] += input[i * cols + j];
    }
  }
}
