extern "C" void conv2d(float* input, float* kernel, float* output) {
    const int batch_size = 32;
    const int input_height = 8;
    const int input_width = 8;
    const int input_channels = 128;
    const int kernel_height = 2;
    const int kernel_width = 2;
    const int output_channels = 64;
    const int stride = 3;
    const int padding = 0;

    const int output_height = (input_height - kernel_height) / stride + 1; 
    const int output_width = (input_width - kernel_width) / stride + 1;

    for (int bs = 0; bs < batch_size; ++bs) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                for (int oc = 0; oc < output_channels; ++oc) {

                    float sum = 0.0f;
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {

                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;

                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {

                                for (int ic = 0; ic < input_channels; ++ic) {
                                    int input_idx = bs * (input_height * input_width * input_channels) +
                                                    ih * (input_width * input_channels) +
                                                    iw * input_channels +
                                                    ic;

                                    int kernel_idx = oc * (kernel_height * kernel_width * input_channels) +
                                                     kh * (kernel_width * input_channels) +
                                                     kw * input_channels +
                                                     ic;

                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }

                    int output_idx = bs * (output_height * output_width * output_channels) +
                                     oh * (output_width * output_channels) +
                                     ow * output_channels +
                                     oc;
                    output[output_idx] = sum;
                }
            }
        }
    }
}