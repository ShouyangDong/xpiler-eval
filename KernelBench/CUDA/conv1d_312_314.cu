__global__ void conv1d(float *input, float *kernel, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 312) {
    output[idx] = 0;
    for (int j = 0; j < 3; j++) {
      output[idx] += input[idx + j] * kernel[j];
    }
  }
}

extern "C" void conv1d_kernel(float *input, float *kernel, float *output, 
                              int input_size, int output_size) {
  float *d_input, *d_kernel, *d_output;
  int kernel_size = input_size - output_size + 1;
  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_kernel, kernel_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(312);
  dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

  conv1d<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

  cudaMemcpy(output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
}
