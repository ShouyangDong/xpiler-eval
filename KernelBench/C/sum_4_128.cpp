extern "C" void sum(float *input, float *output) {
  int rows = 4;
  int cols = 128;
  // sum over cols (dim=1): [M,N] -> [M,1]


  for (int i = 0; i < rows; i++) {
    output[i] = 0.0f;
    for (int j = 0; j < cols; j++) {
      output[i] += input[i * cols + j];
    }
  }
}
