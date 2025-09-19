
extern "C" void max(float *input, float *output) {
  int rows = 16;
  int cols = 32;
  // max over rows (dim=0): [M,N] -> [1,N]

  // Initialize output to -infinity
  for (int j = 0; j < cols; j++) {
    output[j] = -INFINITY;
  }

  // Compare along rows (dim=0)
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (input[i * cols + j] > output[j]) {
        output[j] = input[i * cols + j];
      }
    }
  }
}
