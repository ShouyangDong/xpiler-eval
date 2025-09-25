

__global__ void sum_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= 8)
    return;

  float sum = 0.0f;
  for (int col = 0; col < 64; col++) {
    int idx = row * 64 + col;
    sum += input[idx];
  }
  output[row] = sum;
}

extern "C" void sum_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 8 * 64;
  const int output_size = 8;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(8);
  dim3 numBlocks(1);

  sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);
  hipFree(d_input);
  hipFree(d_output);
}
