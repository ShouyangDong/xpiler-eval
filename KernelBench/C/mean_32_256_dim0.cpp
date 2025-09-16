extern "C" void mean(float *input, float *output) {
  int rows = 32;
  int cols = 256;
  // mean over rows (dim=0): [M,N] -> [1,N]

  // Initialize output to 0
  for (int i = 0; i < 1 * 256; i++) {
    output[i] = 0.0f;
  }

  // Sum over rows
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      output[j] += input[i * cols + j];
    }
    output[j] /= 32;  // divide by number of rows
  }
}
