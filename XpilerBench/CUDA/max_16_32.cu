__global__ void __launch_bounds__(256)
    max_dev(const float *__restrict__ input, float *__restrict__ output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 32)
    return;

  float max_val = -FLT_MAX;
  for (int row = 0; row < 16; row++) {
    int idx = row * 32 + col;
    max_val = fmaxf(max_val, input[idx]);
  }
  output[col] = max_val;
}

extern "C" void max_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 16 * 32;
  const int output_size = 32;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 numBlocks((output_size + 255) / 256);
  max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}
