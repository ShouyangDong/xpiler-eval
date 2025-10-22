#include "hip/hip_runtime.h"


__global__ void max_dev(const float *__restrict__ input,
                        float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_size = 16 * 32;
  if (idx >= output_size)
    return;

  int n = idx / 32;
  int w = idx % 32;

  float max_val = -FLT_MAX;
  for (int h = 0; h < 32; h++) {
    int in_idx = n * (32 * 32) + h * 32 + w;
    max_val = fmaxf(max_val, input[in_idx]);
  }
  output[idx] = max_val;
}

extern "C" void max_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 16 * 32 * 32;
  const int output_size = 16 * 32;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 numBlocks((output_size + 255) / 256);

  max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}