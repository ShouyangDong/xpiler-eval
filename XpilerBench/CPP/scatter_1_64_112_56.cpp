extern "C" void scatter(const float *input, const int *indices, float *output) {
  const int N = 1;
  const int C = 64;
  const int H = 112;
  const int W = 56;
  const int dim = 2; // Scatter along height dimension

  // Initialize output with input values
  for (int i = 0; i < N * C * H * W; i++) {
    output[i] = input[i];
  }

  // For each element in the indices/src tensor
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h_idx = 0; h_idx < H; h_idx++) {
        for (int w = 0; w < W; w++) {
          // Get the target height index from the indices tensor
          int target_h = indices[n * C * H * W + c * H * W + h_idx * W + w];

          // Bounds check
          if (target_h >= 0 && target_h < 112) {
            // Scatter: output[n][c][target_h][w] = input[n][c][h_idx][w]
            output[n * C * H * W + c * H * W + target_h * W + w] =
                input[n * C * H * W + c * H * W + h_idx * W + w];
          }
        }
      }
    }
  }
}
