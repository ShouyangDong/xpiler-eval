extern "C" void sin(float *input, float *output) {
  int rows = 16;
  int cols = 128;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      output[index] = sinf(input[index]);  // 使用单精度 sinf
    }
  }
}
