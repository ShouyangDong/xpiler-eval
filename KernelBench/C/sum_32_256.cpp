extern "C" void sum(float *input, float *output) {
  const int rows = 32;    // axis 0: to be reduced
  const int cols = 256;   // axis 1: preserved

  // Initialize output to 0
  for (int j = 0; j < cols; j++) {
    output[j] = 0.0f;
  }

  // Sum over axis 0 (rows)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[j] += input[i * cols + j];
    }
  }
}
