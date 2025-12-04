

__global__ void max_dev(const float *__restrict__ input,
                        float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 4 * 32)
    return;

  int n = idx / 32;
  int w = idx % 32;

  float max_val = -FLT_MAX;
  for (int h = 0; h < 8; h++) {
    int in_idx = n * (8 * 32) + h * 32 + w;
    max_val = fmaxf(max_val, input[in_idx]);
  }
  output[idx] = max_val;
}

extern "C" void max_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 4 * 8 * 32;
  const int output_size = 4 * 32;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(128);
  dim3 numBlocks((output_size + blockSize.x - 1) / blockSize.x);

  max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}
