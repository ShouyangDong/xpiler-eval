extern "C" void concat(float *input0, float *input1, float *output) {
  int N = 1;
  int C = 128;
  int H = 28;
  int W = 28;
  int axis = 1;  // Concatenate along channel dimension

  int C0 = C;  // Assume input0 has C channels
  int C1 = C;  // Assume input1 has C channels
  int total_C = C0 + C1;

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C0; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int in_idx = n * C0 * H * W + c * H * W + h * W + w;
          int out_idx = n * total_C * H * W + c * H * W + h * W + w;
          output[out_idx] = input0[in_idx];
        }
      }
    }
    for (int c = 0; c < C1; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int in_idx = n * C1 * H * W + c * H * W + h * W + w;
          int out_idx = n * total_C * H * W + (c + C0) * H * W + h * W + w;
          output[out_idx] = input1[in_idx];
        }
      }
    }
  }
}
