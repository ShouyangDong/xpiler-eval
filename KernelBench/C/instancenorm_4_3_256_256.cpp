extern "C" void instancenorm(float *input, float *output) {
  int N = 1, C = 512, H = 7, W = 7;
  int spatial = H * W;
  float eps = 1e-5f;

  float mean[512];
  float var[512];

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      float sum = 0.0f;
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          sum += input[idx];
        }
      }
      mean[c] = sum / spatial;
    }

    for (int c = 0; c < C; c++) {
      float sum_sq = 0.0f;
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          float diff = input[idx] - mean[c];
          sum_sq += diff * diff;
        }
      }
      var[c] = sum_sq / spatial;
    }

    for (int c = 0; c < C; c++) {
      float inv_std = 1.0f / sqrtf(var[c] + eps);
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          output[idx] = (input[idx] - mean[c]) * inv_std;
        }
      }
    }
  }
}
