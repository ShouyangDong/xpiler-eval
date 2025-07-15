__global__ void conv2dnchw(float *input, float *kernel, float *output) {
  int bs = blockIdx.z;  // 批次索引
  int oc = blockIdx.x;  // 输出通道索引
  int oh = threadIdx.y; // 输出高度索引
  int ow = threadIdx.x; // 输出宽度索引

  if (bs < 32 && oc < 64 && oh < 4 && ow < 4) {
    float sum = 0.0;

    for (int ic = 0; ic < 128; ic++) {
      for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
          int ih = oh * 2 + kh;
          int iw = ow * 2 + kw;

          // 输入索引计算
          int input_idx = bs * (128 * 8 * 8) + ic * (8 * 8) + ih * 8 + iw;

          // 卷积核索引计算
          int kernel_idx = oc * (128 * 2 * 2) + ic * (2 * 2) + kh * 2 + kw;

          sum += input[input_idx] * kernel[kernel_idx];
        }
      }
    }

    // 输出索引计算
    int output_idx = bs * (64 * 4 * 4) + oc * (4 * 4) + oh * 4 + ow;

    output[output_idx] = sum;
  }
}

extern "C" void conv2dnchw_kernel(float *input, float *output, float *kernel,
                                  int batch_size, int input_height,
                                  int input_channels, int output_channels,
                                  int kernel_height, int stride) {
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_height - kernel_height) / stride + 1;

  // 计算输入、卷积核和输出的大小
  int input_size = batch_size * input_channels * input_height * input_height;
  int kernel_size =
      output_channels * input_channels * kernel_height * kernel_height;
  int output_size = batch_size * output_channels * output_height * output_width;

  // 分配设备内存
  float *d_input, *d_kernel, *d_output;
  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_kernel, kernel_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  // 复制数据到设备
  hipMemcpy(d_input, input, input_size * sizeof(float),
             hipMemcpyHostToDevice);
  hipMemcpy(d_kernel, kernel, kernel_size * sizeof(float),
             hipMemcpyHostToDevice);

  // 定义块和网格尺寸
  dim3 blockSize(output_width,
                 output_height); // 每个块处理一个输出特征图的空间位置
  dim3 numBlocks(output_channels, 1, batch_size); // 每个输出通道对应一个块

  conv2dnchw<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);



  // 将输出数据复制回主机
  hipMemcpy(output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  // 释放设备和主机内存
  hipFree(d_input);
  hipFree(d_kernel);
  hipFree(d_output);
}
