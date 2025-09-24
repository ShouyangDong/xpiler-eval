

__global__ void mean_kernel_dev(const float *__restrict__ input,
                                float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_size = 8 * 64;
  if (idx >= output_size)
    return;

  int n = idx / 64;
  int w = idx % 64;

  float sum = 0.0f;
  for (int h = 0; h < 32; h++) {

    int in_idx = n * (32 * 64) + h * 64 + w;
    sum += input[in_idx];
  }
  output[idx] = sum / 32.0f;
}

extern "C" void mean_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 8 * 32 * 64;
  const int output_size = 8 * 64;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(512);
  dim3 numBlocks(1);

  mean_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}