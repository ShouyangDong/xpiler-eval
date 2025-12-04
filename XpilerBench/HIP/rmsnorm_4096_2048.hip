

__global__ void rmsnorm(float *A, float *B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1e-5f;

  if (idx < 4096) {

    float sum = 0.0;
    for (int j = 0; j < 2048; j++) {
      sum += A[idx * 2048 + j] * A[idx * 2048 + j];
    }

    float mean = sum / 2048;

    float scale = 1.0 / sqrt(mean + eps);

    for (int j = 0; j < 2048; j++) {
      B[idx * 2048 + j] = A[idx * 2048 + j] * scale;
    }
  }
}

extern "C" void rmsnorm_kernel(float *A, float *B, int size_1, int size_2) {

  float *d_A, *d_B;
  int num_elements = size_1 * size_2;
  hipMalloc(&d_A, num_elements * sizeof(float));
  hipMalloc(&d_B, num_elements * sizeof(float));

  hipMemcpy(d_A, A, num_elements * sizeof(float), hipMemcpyHostToDevice);

  int block_size = 1024;
  int num_blocks = (size_1 + block_size - 1) / block_size;

  rmsnorm<<<num_blocks, block_size>>>(d_A, d_B);

  hipMemcpy(B, d_B, num_elements * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
}
