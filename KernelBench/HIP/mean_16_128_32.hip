

__global__ void mean_kernel_dev(const float *__restrict__ input,
                                float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_size = 16 * 32;
  if (idx >= output_size)
    return;

  int n = idx / 32;
  int w = idx % 32;

  float sum = 0.0f;
  for (int h = 0; h < 128; h++) {

    int in_idx = n * (128 * 32) + h * 32 + w;
    sum += input[in_idx];
  }
  output[idx] = sum / 128.0f;
}

extern "C" void mean_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 16 * 128 * 32;
  const int output_size = 16 * 32;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(512);
  dim3 numBlocks(1);

  mean_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}