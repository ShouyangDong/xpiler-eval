

__global__ void min_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_size = 2 * 5 * 64;
  if (idx >= output_size)
    return;

  int n = idx / (5 * 64);
  int rem = idx % (5 * 64);
  int h = rem / 64;
  int w = rem % 64;

  float min_val = FLT_MAX;
  for (int c = 0; c < 4; c++) {

    int in_idx = n * (4 * 5 * 64) + c * (5 * 64) + h * 64 + w;
    min_val = fminf(min_val, input[in_idx]);
  }
  output[idx] = min_val;
}

extern "C" void min_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 2 * 4 * 5 * 64;
  const int output_size = 2 * 5 * 64;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(640);
  dim3 numBlocks(1);

  min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}