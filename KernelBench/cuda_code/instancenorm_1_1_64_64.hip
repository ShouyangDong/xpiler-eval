#include "hip/hip_runtime.h"


__global__ void __launch_bounds__(256)
    instancenorm(const float *__restrict__ input, float *__restrict__ output,
                 const float *__restrict__ gamma,
                 const float *__restrict__ beta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = 1, C = 1, H = 64, W = 64;
  const int spatial = H * W;
  const int total_elements = N * C * H * W;
  const float eps = 1e-5f;

  if (idx >= total_elements)
    return;

  int n = idx / (C * H * W);
  int c = (idx % (C * H * W)) / (H * W);
  int h = (idx % (H * W)) / W;
  int w = idx % W;

  // Compute mean for current (n, c)
  float sum = 0.0f;
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      int pixel_idx = n * C * H * W + c * H * W + i * W + j;
      sum += input[pixel_idx];
    }
  }
  float mean = sum / spatial;

  // Compute variance for current (n, c)
  float sum_sq = 0.0f;
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      int pixel_idx = n * C * H * W + c * H * W + i * W + j;
      float diff = input[pixel_idx] - mean;
      sum_sq += diff * diff;
    }
  }
  float variance = sum_sq / spatial;

  // Normalize and apply affine transform
  float inv_std = 1.0f / sqrtf(variance + eps);
  float norm_val = (input[idx] - mean) * inv_std;

  float gamma_val = gamma ? gamma[c] : 1.0f;
  float beta_val = beta ? beta[c] : 0.0f;

  output[idx] = norm_val * gamma_val + beta_val;
}

extern "C" void instancenorm_kernel(const float *h_input, float *h_output,
                                    const float *h_gamma, const float *h_beta) {
  float *d_input, *d_output, *d_gamma, *d_beta;
  const int N = 1, C = 1, H = 64, W = 64;
  const int total_elements = N * C * H * W;
  const int param_size = C;

  hipMalloc(&d_input, total_elements * sizeof(float));
  hipMalloc(&d_output, total_elements * sizeof(float));
  hipMalloc(&d_gamma, param_size * sizeof(float));
  hipMalloc(&d_beta, param_size * sizeof(float));

  hipMemcpy(d_input, h_input, total_elements * sizeof(float),
             hipMemcpyHostToDevice);
  if (h_gamma)
    hipMemcpy(d_gamma, h_gamma, param_size * sizeof(float),
               hipMemcpyHostToDevice);
  if (h_beta)
    hipMemcpy(d_beta, h_beta, param_size * sizeof(float),
               hipMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 numBlocks((total_elements + blockSize.x - 1) / blockSize.x);

  instancenorm<<<numBlocks, blockSize>>>(d_input, d_output, d_gamma, d_beta);

  hipMemcpy(h_output, d_output, total_elements * sizeof(float),
             hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
  hipFree(d_gamma);
  hipFree(d_beta);
}
