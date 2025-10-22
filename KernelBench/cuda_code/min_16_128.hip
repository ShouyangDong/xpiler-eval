#include "hip/hip_runtime.h"


__global__ void min_kernel_dev(const float *__restrict__ input,
                               float *__restrict__ output) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= 16)
    return;

  float min_val = FLT_MAX;
  for (int col = 0; col < 128; col++) {
    int idx = row * 128 + col;
    min_val = fminf(min_val, input[idx]);
  }
  output[row] = min_val;
}

extern "C" void min_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 16 * 128;
  const int output_size = 16;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(16);
  dim3 numBlocks(1);

  min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);
  hipFree(d_input);
  hipFree(d_output);
}