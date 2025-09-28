extern "C" void minpool(float *x, float *output) {
  int N = 1;
  int H = 5;
  int W = 5;
  int C = 64;
  int kernel_size = 5;
  int stride = 1;
  int output_H = (H - kernel_size) / stride + 1;
  int output_W = (W - kernel_size) / stride + 1;
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < output_H; h++) {
      for (int w = 0; w < output_W; w++) {
        for (int c = 0; c < C; c++) {
          float min_val = FLT_MAX;
          for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
              int input_h = h * stride + kh;
              int input_w = w * stride + kw;
              if (input_h < H && input_w < W) {
                float val = x[n * H * W * C + (input_h * W + input_w) * C + c];
                if (val < min_val) {
                  min_val = val;
                }
              }
            }
          }
          output[n * output_H * output_W * C + h * output_W * C + w * C + c] =
              min_val;
        }
      }
    }
  }
}
