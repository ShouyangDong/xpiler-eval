
extern "C" void max(float *input, float *output) {
  int rows = 4;
  int cols = 8;
  // max over cols (dim=1): [M,N] -> [M,1]

  // Initialize output to -infinity
  for (int i = 0; i < rows; i++) {
    output[i] = -INFINITY;
  }

  // Compare along cols (dim=1)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (input[i * cols + j] > output[i]) {
        output[i] = input[i * cols + j];
      }
    }
  }
}
