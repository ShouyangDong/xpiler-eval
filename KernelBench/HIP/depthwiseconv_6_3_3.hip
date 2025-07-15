__global__ void depthwiseconv(const float *input, const float *filter,
                                      float *output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z; // 使用 blockIdx.z 处理每个通道的独立计算

  // 检查线程是否在输出范围内
  if (tid_x < 4 && tid_y < 4 && c < 3) {
    // 初始化输出值
    float sum = 0.0;

    // 进行 3x3 滤波器的卷积计算
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int input_x = tid_x + j;
        int input_y = tid_y + i;
        int input_idx = (input_y * 6 + input_x) * 3 + c; // 3 表示通道数
        int filter_idx = (i * 3 + j) * 3 + c;
        sum += input[input_idx] * filter[filter_idx];
      }
    }

    // 将结果存储到输出
    int output_idx = (tid_y * 4 + tid_x) * 3 + c;
    output[output_idx] = sum;
  }
}

extern "C" void depthwiseconv_kernel(float *input, float *kernel, float *output,
                                     int input_height, int kernel_size,
                                     int input_channels) {
  float *d_input, *d_kernel, *d_output;
  int input_size = input_height * input_height * input_channels;
  int output_height = input_height - kernel_size + 1;
  int output_width = input_height - kernel_size + 1;
  int filter_size = kernel_size * kernel_size * input_channels;
  int output_size = output_height * output_width * input_channels;
  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_kernel, filter_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, input, input_size * sizeof(float),
             hipMemcpyHostToDevice);
  hipMemcpy(d_kernel, kernel, filter_size * sizeof(float),
             hipMemcpyHostToDevice);

  // 定义块和网格尺寸
  dim3 blockSize(16, 16);
  dim3 numBlocks((output_width + blockSize.x - 1) / blockSize.x,
                 (output_height + blockSize.y - 1) / blockSize.y,
                 input_channels); // 每个通道使用一个块

depthwiseconv<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);



  hipMemcpy(output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_kernel);
  hipFree(d_output);
}