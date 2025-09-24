

__global__ void min_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_size = 5 * 20;
  if (idx >= output_size)
    return;

  int n = idx / 20;
  int w = idx % 20;

  float min_val = FLT_MAX;
  for (int h = 0; h < 10; h++) {
    int in_idx = n * (10 * 20) + h * 20 + w;
    min_val = fminf(min_val, input[in_idx]);
  }
  output[idx] = min_val;
}

extern "C" void min_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 5 * 10 * 20;
  const int output_size = 5 * 20;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(100);
  dim3 numBlocks(1);

  min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}