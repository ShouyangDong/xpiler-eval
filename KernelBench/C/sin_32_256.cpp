extern "C" void sin_kernel(float *input, float *output) {
  int rows = 32;
  int cols = 256;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      output[index] = sinf(input[index]);  // 使用单精度 sinf
    }
  }
}
