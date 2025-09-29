extern "C" void scatter(const float *input, const int *indices, float *output) {
  const int N = 1;
  const int C = 64;
  const int H = 56;
  const int W = 112;
  const int dim = 3; // Scatter along width dimension

  // Initialize output with input values
  for (int i = 0; i < N * C * H * W; i++) {
    output[i] = input[i];
  }

  // For each element in the indices/src tensor
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w_idx = 0; w_idx < W; w_idx++) {
          // Get the target width index from the indices tensor
          int target_w = indices[n * C * H * W + c * H * W + h * W + w_idx];

          // Bounds check
          if (target_w >= 0 && target_w < 112) {
            // Scatter: output[n][c][h][target_w] = input[n][c][h][w_idx]
            output[n * C * H * W + c * H * W + h * W + target_w] =
                input[n * C * H * W + c * H * W + h * W + w_idx];
          }
        }
      }
    }
  }
}
