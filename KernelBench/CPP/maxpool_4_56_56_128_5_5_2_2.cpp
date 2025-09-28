extern "C" void maxpool(float *x, float *output) {
  int N = 4;
  int H = 56;
  int W = 56;
  int C = 128;
  int kernel_size = 5;
  int stride = 2;
  int output_H = (H - kernel_size) / stride + 1;
  int output_W = (W - kernel_size) / stride + 1;
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < output_H; h++) {
      for (int w = 0; w < output_W; w++) {
        for (int c = 0; c < C; c++) {
          float max_val =
              x[n * H * W * C + h * stride * W * C + w * stride * C + c];
          for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
              int nh = h * stride + kh;
              int nw = w * stride + kw;
              if (nh < H && nw < W) {
                float current_val = x[n * H * W * C + nh * W * C + nw * C + c];
                if (current_val > max_val) {
                  max_val = current_val;
                }
              }
            }
          }
          output[n * output_H * output_W * C + h * output_W * C + w * C + c] =
              max_val;
        }
      }
    }
  }
}