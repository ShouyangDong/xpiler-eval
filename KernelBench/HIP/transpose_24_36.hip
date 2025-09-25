#include <hip_runtime.h>
#include <stdio.h>

__global__ void transpose(const float *__restrict__ input,
                          float *__restrict__ output) {
  int o0 = blockIdx.y * blockDim.y + threadIdx.y;
  int o1 = blockIdx.x * blockDim.x + threadIdx.x;
  int s0_in = 24;
  int s1_in = 36;
  int s0_out = 36;
  int s1_out = 24;
  if (o0 < s0_out && o1 < s1_out) {

    output[o0 * s1_out + o1] = input[o1 * s1_in + o0];
  }
}

extern "C" void transpose_kernel(float *input, float *output, int s0_in,
                                 int s1_in) {

  size_t in_size = s0_in * s1_in * sizeof(float);
  size_t out_size = s0_in * s1_in * sizeof(float);

  float *d_input, *d_output;
  hipMalloc(&d_input, in_size);
  hipMalloc(&d_output, out_size);

  hipMemcpy(d_input, input, in_size, hipMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((s0_in + block.x - 1) / block.x, (s1_in + block.y - 1) / block.y);

  transpose<<<grid, block>>>(d_input, d_output);

  hipMemcpy(output, d_output, out_size, hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}
