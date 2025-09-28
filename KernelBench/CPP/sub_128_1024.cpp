extern "C" void sub(float *input1, float *input2, float *output) {
  int rows = 128;
  int cols = 1024;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      output[index] = input1[index] - input2[index];
    }
  }
}
