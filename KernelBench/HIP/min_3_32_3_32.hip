

__global__ void min_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_size = 3 * 3 * 32;
  if (idx >= output_size)
    return;

  int n = idx / (3 * 32);
  int rem = idx % (3 * 32);
  int h = rem / 32;
  int w = rem % 32;

  float min_val = FLT_MAX;
  for (int c = 0; c < 32; c++) {

    int in_idx = n * (32 * 3 * 32) + c * (3 * 32) + h * 32 + w;
    min_val = fminf(min_val, input[in_idx]);
  }
  output[idx] = min_val;
}

extern "C" void min_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 3 * 32 * 3 * 32;
  const int output_size = 3 * 3 * 32;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
            hipMemcpyHostToDevice);

  dim3 blockSize(288);
  dim3 numBlocks(1);

  min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
            hipMemcpyDeviceToHost);
  hipFree(d_input);
  hipFree(d_output);
}