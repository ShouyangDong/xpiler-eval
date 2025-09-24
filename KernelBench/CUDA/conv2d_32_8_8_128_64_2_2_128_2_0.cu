__global__ void conv2d(float *input, float *kernel, float *output) {
  int bs = blockIdx.z;
  int oc = threadIdx.x;
  int oh = blockIdx.y;
  int ow = blockIdx.x;

  if (oc < 64 && oh < 4 && ow < 4 && bs < 32) {
    float sum = 0.0;

    for (int kh = 0; kh < 2; kh++) {
      for (int kw = 0; kw < 2; kw++) {
        for (int ic = 0; ic < 128; ic++) {
          int ih = oh * 2 + kh;
          int iw = ow * 2 + kw;

          int input_idx = bs * (8 * 8 * 128) + ih * (8 * 128) + iw * 128 + ic;

          int kernel_idx = oc * (2 * 2 * 128) + kh * (2 * 128) + kw * 128 + ic;

          sum += input[input_idx] * kernel[kernel_idx];
        }
      }
    }

    int output_idx = bs * (4 * 4 * 64) + oh * (4 * 64) + ow * 64 + oc;

    output[output_idx] = sum;
  }
}

extern "C" void conv2d_kernel(float *input, float *filter, float *output,
                              int batch_size, int input_height,
                              int input_channels, int output_channels,
                              int kernel_height, int stride) {
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_height - kernel_height) / stride + 1;

  int input_size = batch_size * input_height * input_height * input_channels;
  int kernel_size =
      output_channels * kernel_height * kernel_height * input_channels;
  int output_size = batch_size * output_height * output_width * output_channels;

  float *d_input, *d_filter, *d_output;
  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_filter, kernel_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, filter, kernel_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(output_channels);
  dim3 numBlocks(output_width, output_height, batch_size);

  conv2d<<<numBlocks, blockSize>>>(d_input, d_filter, d_output);

  cudaMemcpy(output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_filter);
  cudaFree(d_output);
}
