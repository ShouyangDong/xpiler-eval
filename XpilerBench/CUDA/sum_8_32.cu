

__global__ void sum_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 32)
    return;

  float sum = 0.0f;
  for (int row = 0; row < 8; row++) {
    int idx = row * 32 + col;
    sum += input[idx];
  }
  output[col] = sum;
}

extern "C" void sum_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 8 * 32;
  const int output_size = 32;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(32);
  dim3 numBlocks(1);

  sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}
