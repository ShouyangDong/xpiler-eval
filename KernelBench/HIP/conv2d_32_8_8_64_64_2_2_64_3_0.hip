__global__ void conv2d(float *input, float *kernel, float *output) {
  int bs = blockIdx.z;  // 批次索引
  int oc = threadIdx.x; // 输出通道索引
  int oh = blockIdx.y;  // 输出高度索引
  int ow = blockIdx.x;  // 输出宽度索引

  if (oc < 64 && oh < 3 && ow < 3 && bs < 32) {
    float sum = 0.0;

    for (int kh = 0; kh < 2; kh++) {
      for (int kw = 0; kw < 2; kw++) {
        for (int ic = 0; ic < 64; ic++) {
          int ih = oh * 3 + kh;
          int iw = ow * 3 + kw;

          int input_idx = bs * (8 * 8 * 64) + ih * (8 * 64) + iw * 64 + ic;

          int kernel_idx = oc * (2 * 2 * 64) + kh * (2 * 64) + kw * 64 + ic;

          sum += input[input_idx] * kernel[kernel_idx];
        }
      }
    }

    int output_idx = bs * (3 * 3 * 64) + oh * (3 * 64) + ow * 64 + oc;

    output[output_idx] = sum;
  }
}

extern "C" void conv2d_kernel(float *input, float *output, float *filter,
                              int batch_size, int input_height,
                              int input_channels, int output_channels,
                              int kernel_height, int stride) {
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_height - kernel_height) / stride + 1;

  // 定义输入尺寸和卷积核尺寸
  int input_size = batch_size * input_height * input_height * input_channels;
  int kernel_size =
      output_channels * kernel_height * kernel_height * input_channels;
  int output_size = batch_size * output_height * output_width * output_channels;

  // 分配设备内存
  float *d_input, *d_filter, *d_output;
  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_filter, kernel_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  // 复制数据到设备
  hipMemcpy(d_input, input, input_size * sizeof(float),
             hipMemcpyHostToDevice);
  hipMemcpy(d_filter, filter, kernel_size * sizeof(float),
             hipMemcpyHostToDevice);

  // 定义块和网格尺寸
  dim3 blockSize(output_channels); // 每个块中有 output_channels 个线程
  dim3 numBlocks(output_width, output_height,
                 batch_size); // 输出的宽、高以及批次

  conv2d<<<numBlocks, blockSize>>>(d_input, d_filter, d_output);

  // 将输出数据复制回主机
  hipMemcpy(output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  // 释放设备和主机内存
  hipFree(d_input);
  hipFree(d_filter);
  hipFree(d_output);
}
