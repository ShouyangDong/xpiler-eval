

__global__ void sum_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 128 * 128)
    return;

  float sum = 0.0f;
  for (int i = 0; i < 16; i++) {
    int input_idx = i * (128 * 128) + idx;
    sum += input[input_idx];
  }
  output[idx] = sum;
}

extern "C" void sum_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 16 * 128 * 128;
  const int output_size = 128 * 128;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
            hipMemcpyHostToDevice);

  int total_elements = 128 * 128;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;

  sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
            hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}
