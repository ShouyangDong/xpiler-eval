
extern "C" void batchnorm(float *input, float *output) {
  int N = 1, C = 4, H = 32, W = 32;
  int spatial = H * W;
  int size_per_channel = N * H * W;  // usually N=1
  float eps = 1e-5f;

  // Temporary storage for mean and var per channel
  float mean[512];   // 支持最多 512 通道
  float var[512];

  // Step 1: Compute mean for each channel
  for (int c = 0; c < C; c++) {
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          sum += input[idx];
        }
      }
    }
    mean[c] = sum / (N * spatial);
  }

  // Step 2: Compute variance for each channel
  for (int c = 0; c < C; c++) {
    float sum_sq = 0.0f;
    for (int n = 0; n < N; n++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          float diff = input[idx] - mean[c];
          sum_sq += diff * diff;
        }
      }
    }
    var[c] = sum_sq / (N * spatial);
  }

  // Step 3: Normalize (gamma=1, beta=0)
  for (int c = 0; c < C; c++) {
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    for (int n = 0; n < N; n++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          output[idx] = (input[idx] - mean[c]) * inv_std;
        }
      }
    }
  }
}
