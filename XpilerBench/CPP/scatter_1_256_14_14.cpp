extern "C" void scatter(const float *input, const int *indices, float *output) {
  const int N = 1;
  const int C = 256;
  const int H = 14;
  const int W = 14;
  const int dim = 1; // Scatter along channel dimension

  // Initialize output with input values
  for (int i = 0; i < N * 256 * H * W; i++) {
    output[i] = input[i];
  }

  // For each element in the indices/src tensor
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c_idx = 0; c_idx < C; c_idx++) {
          // Get the target channel index from the indices tensor
          int target_c = indices[n * C * H * W + c_idx * H * W + h * W + w];

          // Bounds check
          if (target_c >= 0 && target_c < 256) {
            // Scatter: output[n][target_c][h][w] = input[n][c_idx][h][w]
            output[n * 256 * H * W + target_c * H * W + h * W + w] =
                input[n * C * H * W + c_idx * H * W + h * W + w];
          }
        }
      }
    }
  }
}
