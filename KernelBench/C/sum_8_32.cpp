extern "C" void sum(float *input, float *output) {
  int rows = 8;
  int cols = 32;
  // sum over rows (dim=0): [M,N] -> [1,N]

  // Initialize output to 0
  for (int i = 0; i < 1 * 32; i++) {
    output[i] = 0.0f;
  }

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      output[j] += input[i * cols + j];
    }
  }
}
