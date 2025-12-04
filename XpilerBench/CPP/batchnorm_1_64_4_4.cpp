extern "C" void batchnorm(float *input, float *output,
                          float *mean,   // [C]
                          float *var,    // [C]
                          float *weight, // [C] gamma
                          float *bias,   // [C] beta
                          float eps) {
  int N = 1, C = 64, H = 4, W = 4;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      float inv_std = 1.0f / sqrtf(var[c] + eps);
      float gamma = weight[c];
      float beta = bias[c];
      float m = mean[c];

      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          float normalized = (input[idx] - m) * inv_std;
          output[idx] = gamma * normalized + beta;
        }
      }
    }
  }
}
