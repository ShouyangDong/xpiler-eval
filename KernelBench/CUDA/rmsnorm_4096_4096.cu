

__global__ void rmsnorm(float *A, float *B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1e-5f;

  if (idx < 4096) {
    // Calculate sum
    float sum = 0.0;
    for (int j = 0; j < 4096; j++) {
      sum += A[idx * 4096 + j] * A[idx * 4096 + j];
    }

    // Calculate mean
    float mean = sum / 4096;

    // Calculate scale
    float scale = 1.0 / sqrt(mean + eps);

    // Normalize and store in B
    for (int j = 0; j < 4096; j++) {
      B[idx * 4096 + j] = A[idx * 4096 + j] * scale;
    }
  }
}

extern "C" void rmsnorm_kernel(float *A, float *B, int size_1, int size_2) {
  // Allocate memory on the device
  float *d_A, *d_B;
  int num_elements = size_1 * size_2;
  cudaMalloc(&d_A, num_elements * sizeof(float));
  cudaMalloc(&d_B, num_elements * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, A, num_elements * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int block_size = 1024;
  int num_blocks = (size_1 + block_size - 1) / block_size;

  // Launch kernel
  rmsnorm<<<num_blocks, block_size>>>(d_A, d_B);

  // Copy the result back to host
  cudaMemcpy(B, d_B, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
}
