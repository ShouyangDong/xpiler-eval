extern "C" void sumpool(float *x, float *output) {
  int N = 1;
  int H = 5;
  int W = 5;
  int C = 64;
  int kernel_size = 3;
  int stride = 2;
  int output_H = (H - kernel_size) / stride + 1;
  int output_W = (W - kernel_size) / stride + 1;
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < output_H; h++) {
      for (int w = 0; w < output_W; w++) {
        for (int c = 0; c < C; c++) {
          float sum = 0.0;
          for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
              int nh = h * stride + kh;
              int nw = w * stride + kw;
              if (nh < H && nw < W) {
                sum += x[n * H * W * C + nh * W * C + nw * C + c];
              }
            }
          }
          output[n * output_H * output_W * C + h * output_W * C + w * C + c] =
              sum;
        }
      }
    }
  }
}
