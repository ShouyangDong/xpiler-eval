#include "hip/hip_runtime.h"
__global__ void max_dev(const float *__restrict__ input,
                        float *__restrict__ output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 8)
    return;

  float max_val = -FLT_MAX;
  for (int n = 0; n < 4; n++) {
    int idx = n * 8 + col;
    max_val = fmaxf(max_val, input[idx]);
  }
  output[col] = max_val;
}

extern "C" void max_kernel(const float *h_input, float *h_output) {
  float *d_input, *d_output;
  const int input_size = 4 * 8;
  const int output_size = 8;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, output_size * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(8);
  dim3 numBlocks(1);

  max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

  hipMemcpy(h_output, d_output, output_size * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}
