extern "C" void concat(float *input0, float *input1, float *output) {
  int N = 1;
  int C = 64;
  int H = 112;
  int W = 56;
  int axis = 2; // Concatenate along height dimension

  int H0 = H; // Assume input0 has H elements along axis 2
  int H1 = H; // Assume input1 has H elements along axis 2
  int total_H = H0 + H1;

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H0; h++) {
        for (int w = 0; w < W; w++) {
          int in_idx = n * C * H0 * W + c * H0 * W + h * W + w;
          int out_idx = n * C * total_H * W + c * total_H * W + h * W + w;
          output[out_idx] = input0[in_idx];
        }
      }
      for (int h = 0; h < H1; h++) {
        for (int w = 0; w < W; w++) {
          int in_idx = n * C * H1 * W + c * H1 * W + h * W + w;
          int out_idx =
              n * C * total_H * W + c * total_H * W + (h + H0) * W + w;
          output[out_idx] = input1[in_idx];
        }
      }
    }
  }
}