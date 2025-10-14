extern "C" void mean(float *input, float *output) {
  const int N = 16;
  const int C = 128;
  const int D = 32;

  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      output[n * 1 * D + 0 * D + d] = 0.0f;
    }
  }
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int d = 0; d < D; d++) {
        int input_idx = n * C * D + c * D + d;
        int output_idx = n * 1 * D + 0 * D + d;

        output[output_idx] += input[input_idx];
      }
    }
  }

  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      int output_idx = n * 1 * D + 0 * D + d;
      output[output_idx] /= C;
    }
  }
}