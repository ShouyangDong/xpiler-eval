extern "C" void mean(float *input, float *output) {
  int rows = 8;
  int cols = 32;
  // mean over cols (dim=1): [M,N] -> [M,1]

  // Initialize output to 0
  for (int i = 0; i < 8 * 1; i++) {
    output[i] = 0.0f;
  }

  // Sum over cols
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[i] += input[i * cols + j];
    }
    output[i] /= 32;  // divide by number of cols
  }
}
