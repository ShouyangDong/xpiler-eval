
extern "C" void max(float *input, float *output) {
  int rows = 128;
  int cols = 256;

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
