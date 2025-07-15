__global__ void mha(float *Q, float *K, float *V,
                                             float *output) {

  __shared__ float score[36]; // 使用共享内存存储 score, 大小为 heads * heads
  float scaling_factor = 1.0f / sqrtf((float)256);
  int i = blockIdx.x;  // batch index
  int j = blockIdx.y;  // query index within sequence length
  int m = threadIdx.x; // head index

  for (int n = 0; n < 6; n++) {
    score[m * 6 + n] = 0.0;
    for (int p = 0; p < 256; p++) {
      score[m * 6 + n] += Q[i * 4096 * 6 * 256 + j * 6 * 256 + m * 256 + p] *
                          K[i * 4096 * 6 * 256 + j * 6 * 256 + n * 256 + p];
    }
  }

  // score
  for (int n_sc = 0; n_sc < 6; n_sc++) {
    score[m * 6 + n_sc] = score[m * 6 + n_sc] * scaling_factor;
  }

  float sum = 0;

  for (int i_ex = 0; i_ex < 6; ++i_ex) {
    score[m * 6 + i_ex] = expf(score[m * 6 + i_ex]);
  }
  for (int i_sf = 0; i_sf < 6; ++i_sf) {
    sum += score[m * 6 + i_sf];
  }
  for (int k_sf = 0; k_sf < 6; ++k_sf) {
    score[m * 6 + k_sf] = score[m * 6 + k_sf] / sum;
  }

  // The final Matmul
  for (int n_fl = 0; n_fl < 256; ++n_fl) {
    output[i * 4096 * 6 * 256 + j * 6 * 256 + m * 256 + n_fl] = 0.0;
    for (int k_fl = 0; k_fl < 6; ++k_fl) {
      output[i * 4096 * 6 * 256 + j * 6 * 256 + m * 256 + n_fl] +=
          score[m * 6 + k_fl] *
          V[i * 4096 * 6 * 256 + j * 6 * 256 + k_fl * 256 + n_fl];
    }
  }
}

extern "C" void mha_kernel(float *queries, float *keys, float *values,
                           float *output, int batch_size, int seq_len,
                           int num_heads, int head_dim) {

  int size = batch_size * seq_len * num_heads * head_dim;
  float *d_queries, *d_keys, *d_values, *d_output;
  hipMalloc(&d_queries, size * sizeof(float));
  hipMalloc(&d_keys, size * sizeof(float));
  hipMalloc(&d_values, size * sizeof(float));
  hipMalloc(&d_output, size * sizeof(float));

  hipMemcpy(d_queries, queries, size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_keys, keys, size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_values, values, size * sizeof(float), hipMemcpyHostToDevice);

  dim3 grid(batch_size, seq_len);
  dim3 block(num_heads);

  mha<<<grid, block>>>(d_queries, d_keys, d_values,d_output);

  hipMemcpy(output, d_output, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_queries);
  hipFree(d_keys);
  hipFree(d_values);
  hipFree(d_output);
}
