

__global__ void mean_kernel_dev(const float *__restrict__ input,
                                float *__restrict__ output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 128)
    return;

  float sum = 0.0f;
  for (int row = 0; row < 16; row++) {
    int idx = row * 128 + col;
    sum += input[idx];
  }
  output[col] = sum / 16.0f;
}

extern "C" void mean_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 16 * 128;
  const int output_size = 128;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
            hipMemcpyHostToDevice);

  dim3 blockSize(128);
  dim3 numBlocks(1);

  mean_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
            hipMemcpyDeviceToHost);
  hipFree(d_input);
  hipFree(d_output);
}