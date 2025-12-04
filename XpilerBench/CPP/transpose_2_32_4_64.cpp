extern "C" void transpose(float *input, float *output) {
  const int N = 2;
  const int C = 32;
  const int H = 4;
  const int W = 64;

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int in_idx = n * (C * H * W) + c * (H * W) + h * W + w;
          int out_idx = n * (H * W * C) + h * (W * C) + w * C + c;

          output[out_idx] = input[in_idx];
        }
      }
    }
  }
}
