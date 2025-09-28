extern "C" void concat(float *input0, float *input1, float *output) {
  int N = 8;
  int C = 768;
  int H = 1;
  int W = 1;
  int axis = 1;  // Concatenate along channel dimension

  int C0 = C;  // Assume input0 has C channels
  int C1 = C;  // Assume input1 has C channels
  int total_C = C0 + C1;

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C0; c++) {
      int in_idx = n * C0 * H * W + c * H * W;
      int out_idx = n * total_C * H * W + c * H * W;
      output[out_idx] = input0[in_idx];
    }
    for (int c = 0; c < C1; c++) {
      int in_idx = n * C1 * H * W + c * H * W;
      int out_idx = n * total_C * H * W + (c + C0) * H * W;
      output[out_idx] = input1[in_idx];
    }
  }
}
