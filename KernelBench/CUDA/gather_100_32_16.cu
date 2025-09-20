// ============================================================ //
// 实例 4: (100, 32), indices=16
// ============================================================ //

__global__ void gather(const float* params,
                                     const int* indices,
                                     float* output) {
  constexpr int PARAMS_BATCH = 100;
  constexpr int PARAMS_LEN   = 32;
  constexpr int INDICES_LEN  = 16;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= INDICES_LEN) return;

  int idx = indices[i];
  float* out_row = &output[i * PARAMS_LEN];

  if (idx < 0 || idx >= PARAMS_BATCH) {
    // 越界：置零
    for (int j = 0; j < PARAMS_LEN; ++j) {
      out_row[j] = 0.0f;
    }
  } else {
    const float* src_row = &params[idx * PARAMS_LEN];
    for (int j = 0; j < PARAMS_LEN; ++j) {
      out_row[j] = src_row[j];
    }
  }
}


extern "C" void gather_kernl(const float* d_params,
                                     const int* d_indices,
                                     float* d_output,
                                     int size1,
                                     int size2,
                                     int size3) {

  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc(&d_A, size1 * sizeof(float));
  cudaMalloc(&d_B, size2 * sizeof(int));
  cudaMalloc(&d_C, size3 * sizeof(float));

  cudaMemcpy(d_A, d_params, size1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, d_indices, size2 * sizeof(int), cudaMemcpyHostToDevice);

  constexpr int block_size = 16;
  constexpr int grid_size  = 1;

  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output);

  cudaMemcpy(d_output, d_C, size3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}