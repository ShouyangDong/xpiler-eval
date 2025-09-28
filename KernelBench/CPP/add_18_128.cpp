extern "C" void add(float *input1, float *input2, float *output) {
  int rows = 18;
  int cols = 128;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      output[index] = input1[index] + input2[index];
    }
  }
}
