
extern "C" void conv2d(float *input, float *kernel, float *output) {
  int batch_size = 16;
  int input_height = 8;
  int input_width = 8;
  int input_channels = 64;
  int kernel_height = 2;
  int kernel_width = 2;
  int output_channels = 128;
  int stride = 3;
  int padding = 0;
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_width - kernel_width) / stride + 1;

  for (int bs = 0; bs < batch_size; bs++) {
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        for (int oc = 0; oc < output_channels; oc++) {
          float sum = 0.0;
          for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
              for (int ic = 0; ic < input_channels; ic++) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;
                sum +=
                    input[bs * input_height * input_width * input_channels +
                          ih * input_width * input_channels +
                          iw * input_channels + ic] *
                    kernel[oc * kernel_height * kernel_width * input_channels +
                           kh * kernel_width * input_channels +
                           kw * input_channels + ic];
              }
            }
          }
          output[bs * output_height * output_width * output_channels +
                 oh * output_width * output_channels + ow * output_channels +
                 oc] = sum;
        }
      }
    }
  }
}