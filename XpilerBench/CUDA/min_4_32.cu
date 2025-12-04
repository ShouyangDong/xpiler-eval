

__global__ void min_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= 4)
    return;

  float min_val = FLT_MAX;
  for (int col = 0; col < 32; col++) {
    int idx = row * 32 + col;
    min_val = fminf(min_val, input[idx]);
  }
  output[row] = min_val;
}

extern "C" void min_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 4 * 32;
  const int output_size = 4;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(4);
  dim3 numBlocks(1);

  min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}