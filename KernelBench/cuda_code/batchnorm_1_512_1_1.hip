#include "hip/hip_runtime.h"
__global__ void __launch_bounds__(256)
    batchnorm(const float *__restrict__ input, float *__restrict__ output,
              const float *__restrict__ mean,
              const float *__restrict__ variance,
              const float *__restrict__ gamma, const float *__restrict__ beta) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 512)
    return;

  int channel_idx = idx;

  float eps = 1e-5f;
  float std = sqrtf(variance[channel_idx] + eps);
  output[idx] = gamma[channel_idx] * (input[idx] - mean[channel_idx]) / std +
                beta[channel_idx];
}

extern "C" void batchnorm_kernel(const float *h_input, float *h_output,
                                 const float *h_mean, const float *h_variance,
                                 const float *h_gamma, const float *h_beta) {
  float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;

  const int input_size = 512;
  const int num_channels = 512;

  hipMalloc(&d_input, input_size * sizeof(float));
  hipMalloc(&d_output, input_size * sizeof(float));
  hipMalloc(&d_mean, num_channels * sizeof(float));
  hipMalloc(&d_variance, num_channels * sizeof(float));
  hipMalloc(&d_gamma, num_channels * sizeof(float));
  hipMalloc(&d_beta, num_channels * sizeof(float));

  hipMemcpy(d_input, h_input, input_size * sizeof(float),
             hipMemcpyHostToDevice);
  hipMemcpy(d_mean, h_mean, num_channels * sizeof(float),
             hipMemcpyHostToDevice);
  hipMemcpy(d_variance, h_variance, num_channels * sizeof(float),
             hipMemcpyHostToDevice);
  hipMemcpy(d_gamma, h_gamma, num_channels * sizeof(float),
             hipMemcpyHostToDevice);
  hipMemcpy(d_beta, h_beta, num_channels * sizeof(float),
             hipMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 numBlocks((input_size + 255) / 256);

  batchnorm<<<numBlocks, blockSize>>>(d_input, d_output, d_mean, d_variance,
                                      d_gamma, d_beta);

  hipMemcpy(h_output, d_output, input_size * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
  hipFree(d_mean);
  hipFree(d_variance);
  hipFree(d_gamma);
  hipFree(d_beta);
}
