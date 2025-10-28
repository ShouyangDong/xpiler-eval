#include "hip/hip_runtime.h"


__global__ void __launch_bounds__(256)
    max_dev(const float *__restrict__ input, float *__restrict__ output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 128)
    return;

  float max_val = -FLT_MAX;
  for (int row = 0; row < 64; row++) {
    int idx = row * 128 + col;
    float val = input[idx];
    max_val = fmaxf(max_val, val);
  }
  output[col] = max_val;
}

extern "C" void max_kernel(float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 8192;
  const int output_size = 128;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 numBlocks((128 + 255) / 256);

  max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}
