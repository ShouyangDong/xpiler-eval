extern "C" void sum(float *input, float *output) {
  int rows = 64;
  int cols = 512;
  // sum over cols (dim=1): [M,N] -> [M,1]

  // Initialize output to 0
  for (int i = 0; i < 64 * 1; i++) {
    output[i] = 0.0f;
  }

  for (int i = 0; i < rows; i++) {
    output[i] = 0;
    for (int j = 0; j < cols; j++) {
      output[i] += input[i * cols + j];
    }
  }
}
