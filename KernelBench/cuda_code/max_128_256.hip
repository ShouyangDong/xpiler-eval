#include "hip/hip_runtime.h"


__global__ void max_dev(const float *__restrict__ input,
                        float *__restrict__ output) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= 128)
    return;

  float max_val = -FLT_MAX;
  for (int col = 0; col < 256; col++) {
    int idx = row * 256 + col;
    max_val = fmaxf(max_val, input[idx]);
  }
  output[row] = max_val;
}

extern "C" void max_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 128 * 256;
  const int output_size = 128;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(128);
  dim3 numBlocks(1);

  max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}