__global__ void __launch_bounds__(256)
    batchnorm(const float *__restrict__ input, float *__restrict__ output,
              const float *__restrict__ mean,
              const float *__restrict__ variance,
              const float *__restrict__ gamma, const float *__restrict__ beta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1250)
    return;

  int tmp_idx = idx;
  int offsets[4];
  for (int i = 0; i < 4; ++i) {
    offsets[4 - i - 1] = tmp_idx % 10;
    tmp_idx /= 10;
  }

  int channel_idx = offsets[1];

  output[idx] = gamma[channel_idx] * (input[idx] - mean[channel_idx]) /
                    sqrt(variance[channel_idx] + 1e-5f) +
                beta[channel_idx];
}

extern "C" void batchnorm_kernel(const float *h_input, float *h_output,
                                 const float *h_mean, const float *h_variance,
                                 const float *h_gamma, const float *h_beta) {
  float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;
  const int input_size = 16 * 3 * 32 * 32;

  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, input_size * sizeof(float));
  cudaMalloc(&d_mean, 3 * sizeof(float));
  cudaMalloc(&d_variance, 3 * sizeof(float));
  cudaMalloc(&d_gamma, 3 * sizeof(float));
  cudaMalloc(&d_beta, 3 * sizeof(float));

  cudaMemcpy(d_input, h_input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean, h_mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_variance, h_variance, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, h_beta, 3 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 numBlocks((input_size + 255) / 256);

  batchnorm<<<numBlocks, blockSize>>>(d_input, d_output, d_mean, d_variance,
                                      d_gamma, d_beta);

  cudaMemcpy(h_output, d_output, input_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_mean);
  cudaFree(d_variance);
  cudaFree(d_gamma);
  cudaFree(d_beta);
}
