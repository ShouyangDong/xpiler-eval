extern "C" void concat(float *input0, float *input1, float *output) {
  int N = 1;
  int C = 64;
  int H = 56;
  int W = 112;
  int axis = 3;  // Concatenate along width dimension

  int W0 = W;  // Assume input0 has W elements along axis 3
  int W1 = W;  // Assume input1 has W elements along axis 3
  int total_W = W0 + W1;

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W0; w++) {
          int in_idx = n * C * H * W0 + c * H * W0 + h * W0 + w;
          int out_idx = n * C * H * total_W + c * H * total_W + h * total_W + w;
          output[out_idx] = input0[in_idx];
        }
        for (int w = 0; w < W1; w++) {
          int in_idx = n * C * H * W1 + c * H * W1 + h * W1 + w;
          int out_idx = n * C * H * total_W + c * H * total_W + h * total_W + (w + W0);
          output[out_idx] = input1[in_idx];
        }
      }
    }
  }
}