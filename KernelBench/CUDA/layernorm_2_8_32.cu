__global__ void layernorm(float *A, float *gamma, float *beta, float *B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 16) {
    float mean = 0.0;
    float variance = 0.0;
    float diff[32];
    // Calculate mean
    for (int i_mean = 0; i_mean < 32; i_mean++) {
      mean += A[idx * 32 + i_mean];
    }
    mean /= 32;
    // Calculate variance
    for (int i_diff = 0; i_diff < 32; i_diff++) {
      diff[i_diff] = A[idx * 32 + i_diff] - mean;
    }

    for (int i_pow = 0; i_pow < 32; i_pow++) {
      diff[i_pow] = diff[i_pow] * diff[i_pow];
    }
    for (int i_var = 0; i_var < 32; i_var++) {
      variance += diff[i_var];
    }
    variance = sqrt(variance / 32);

    // Normalize A
    for (int i_norm = 0; i_norm < 32; i_norm++) {
      diff[i_norm] = (A[idx * 32 + i_norm] - mean);
    }

    for (int i_mul = 0; i_mul < 32; i_mul++) {
      diff[i_mul] = diff[i_mul] * gamma[i_mul];
    }

    for (int i_div = 0; i_div < 32; i_div++) {
      diff[i_div] = diff[i_div] / (variance + 1e-5f);
    }

    for (int i_bet = 0; i_bet < 32; i_bet++) {
      B[idx * 32 + i_bet] = diff[i_bet] + beta[i_bet];
    }
  }
}

extern "C" void layernorm_kernel(float *A, float *gamma, float *beta, float *B,
                                 int batch_size, int seq_length, int d_model) {
  // Allocate memory on the device
  float *d_A, *d_B, *d_gamma, *d_beta;
  int num_elements = batch_size * seq_length * d_model;
  cudaMalloc(&d_A, num_elements * sizeof(float));
  cudaMalloc(&d_B, num_elements * sizeof(float));
  cudaMalloc(&d_gamma, d_model * sizeof(float));
  cudaMalloc(&d_beta, d_model * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, A, num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta, d_model * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int block_size = 16;
  int num_blocks = (batch_size * seq_length + block_size - 1) / block_size;

  layernorm<<<num_blocks, block_size>>>(d_A, d_gamma, d_beta, d_B);
  // Copy the result back to host
  cudaMemcpy(B, d_B, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_gamma);
  cudaFree(d_beta);
}
