
extern "C" void max(float *input, float *output) {
  int rows = 4;
  int cols = 8;

  for (int i = 0; i < rows; i++) {
    output[i] = -INFINITY;
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (input[i * cols + j] > output[i]) {
        output[i] = input[i * cols + j];
      }
    }
  }
}
