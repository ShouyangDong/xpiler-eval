__global__ void conv1d(float *input, float *kernel, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 25) {
    output[idx] = 0;
    for (int j = 0; j < 3; j++) {
      output[idx] += input[idx + j] * kernel[j];
    }
  }
}

extern "C" void conv1d_kernel(float *input, float *output, float *kernel,
                              int input_size, int output_size) {
  float *d_input, *d_kernel, *d_output;
  int kernel_size = input_size - output_size + 1;
  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_kernel, kernel_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, input, input_size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_kernel, kernel, kernel_size * sizeof(float),
            hipMemcpyHostToDevice);

  dim3 blockSize(25);
  dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

  conv1d<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

  hipMemcpy(output, d_output, output_size * sizeof(float),
            hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_kernel);
  hipFree(d_output);
}
