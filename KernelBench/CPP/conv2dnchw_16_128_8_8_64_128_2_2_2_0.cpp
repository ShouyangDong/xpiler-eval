extern "C" void conv2dnchw(float *input, float *kernel, float *output) {
    const int batch_size = 16;
    const int input_height = 8;
    const int input_width = 8;
    const int input_channels = 128;
    const int kernel_height = 2;
    const int kernel_width = 2;
    const int output_channels = 64;
    const int stride = 2;
    const int padding = 0;

    const int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    for (int bs = 0; bs < batch_size; ++bs) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {

                    float sum = 0.0f;

                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {

                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    float input_val =
                                        input[bs * input_channels * input_height * input_width +
                                              ic * input_height * input_width +
                                              ih * input_width + iw];

                                    float kernel_val =
                                        kernel[oc * input_channels * kernel_height * kernel_width +
                                               ic * kernel_height * kernel_width +
                                               kh * kernel_width + kw];

                                    sum += input_val * kernel_val;
                                }
                            }
                        }
                    }

                    output[bs * output_channels * output_height * output_width +
                           oc * output_height * output_width +
                           oh * output_width + ow] = sum;
                }
            }
        }
    }
}