extern "C" void mean(float *input, float *output) {
  int rows = 1;
  int cols = 64;
  // mean over cols (dim=1): [M,N] -> [M,1]

  // Sum over cols
  for (int i = 0; i < rows; i++) {
    output[i] = 0.0f;
    for (int j = 0; j < cols; j++) {
      output[i] += input[i * cols + j];
    }
    output[i] /= 64;  // divide by number of cols
  }
}
