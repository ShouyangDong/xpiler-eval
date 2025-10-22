#include "hip/hip_runtime.h"


__global__ void mean_kernel_dev(const float *__restrict__ input,
                                float *__restrict__ output) {
  int tid = threadIdx.x;
  if (tid != 0)
    return;

  float sum = 0.0f;
  for (int col = 0; col < 64; col++) {
    int idx = 0 * 64 + col;
    sum += input[idx];
  }
  output[0] = sum / 64.0f;
}

extern "C" void mean_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 1 * 64;
  const int output_size = 1;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(1);
  dim3 numBlocks(1);

  mean_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);
  hipFree(d_input);
  hipFree(d_output);
}