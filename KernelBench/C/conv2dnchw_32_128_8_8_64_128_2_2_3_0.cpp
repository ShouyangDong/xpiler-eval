
extern "C" void conv2dnchw(float *input, float *kernel, float *output) {
  int batch_size = 32;
  int input_height = 8;
  int input_width = 8;
  int input_channels = 128;
  int kernel_height = 2;
  int kernel_width = 2;
  int output_channels = 64;
  int stride = 3;
  int padding = 0;
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_width - kernel_width) / stride + 1;

  for (int bs = 0; bs < batch_size; bs++) {
    for (int oc = 0; oc < output_channels; oc++) {
      for (int oh = 0; oh < output_height; oh++) {
        for (int ow = 0; ow < output_width; ow++) {
          float sum = 0.0;
          for (int ic = 0; ic < input_channels; ic++) {
            for (int kh = 0; kh < kernel_height; kh++) {
              for (int kw = 0; kw < kernel_width; kw++) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;
                sum +=
                    input[bs * input_channels * input_height * input_width +
                          ic * input_height * input_width + ih * input_width +
                          iw] *
                    kernel[oc * kernel_height * kernel_width * input_channels +
                           ic * kernel_height * kernel_width +
                           kh * kernel_width + kw];
              }
            }
          }
          output[bs * output_channels * output_height * output_width +
                 oc * output_height * output_width + oh * output_width + ow] =
              sum;
        }
      }
    }
  }
}