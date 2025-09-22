__global__ void __launch_bounds__(256)
batchnorm(const float *__restrict__ input, float *__restrict__ output, 
          const float *__restrict__ mean, const float *__restrict__ variance,
          const float *__restrict__ gamma, const float *__restrict__ beta) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 512) return;  // total elements = 1*512*1*1 = 512

    // For input shape [1, 512, 1, 1], the index `idx` directly corresponds to channel
    // Because H=1, W=1, so each channel has only one element
    int channel_idx = idx;  // since each channel has one element, idx == channel_idx

    // Apply Batch Normalization
    float eps = 1e-5f;
    float std = sqrtf(variance[channel_idx] + eps);
    output[idx] = gamma[channel_idx] * (input[idx] - mean[channel_idx]) / std + beta[channel_idx];
}

extern "C" void batchnorm_kernel(const float *h_input, float *h_output,
                                 const float *h_mean, const float *h_variance,
                                 const float *h_gamma, const float *h_beta) {
    float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;

    const int input_size = 512;     // 1 * 512 * 1 * 1
    const int num_channels = 512;   // C = 512

    // Allocate device memory
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    cudaMalloc(&d_mean, num_channels * sizeof(float));
    cudaMalloc(&d_variance, num_channels * sizeof(float));
    cudaMalloc(&d_gamma, num_channels * sizeof(float));
    cudaMalloc(&d_beta, num_channels * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, num_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, h_variance, num_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, num_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, num_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(256);
    dim3 numBlocks((input_size + 255) / 256);  // (512 + 255) / 256 = 2 blocks

    batchnorm<<<numBlocks, blockSize>>>(d_input, d_output, d_mean, d_variance, d_gamma, d_beta);
    // Copy result back
    cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}
