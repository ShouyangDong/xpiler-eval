
extern "C" void max(float *input, float *output) {
  int rows = 512;
  int cols = 1024;

  for (int j = 0; j < cols; j++) {
    output[j] = -INFINITY;
  }

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (input[i * cols + j] > output[j]) {
        output[j] = input[i * cols + j];
      }
    }
  }
}
