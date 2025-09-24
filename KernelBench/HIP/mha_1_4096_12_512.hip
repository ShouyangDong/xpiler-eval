__global__ void mha(float *Q, float *K, float *V, float *output) {

  __shared__ float score[144];
  float scaling_factor = 1.0f / sqrtf((float)512);
  int i = blockIdx.x;
  int j = blockIdx.y;
  int m = threadIdx.x;

  for (int n = 0; n < 12; n++) {
    score[m * 12 + n] = 0.0;
    for (int p = 0; p < 512; p++) {
      score[m * 12 + n] += Q[i * 4096 * 12 * 512 + j * 12 * 512 + m * 512 + p] *
                           K[i * 4096 * 12 * 512 + j * 12 * 512 + n * 512 + p];
    }
  }

  for (int n_sc = 0; n_sc < 12; n_sc++) {
    score[m * 12 + n_sc] = score[m * 12 + n_sc] * scaling_factor;
  }

  float sum = 0;

  for (int i_ex = 0; i_ex < 12; ++i_ex) {
    score[m * 12 + i_ex] = expf(score[m * 12 + i_ex]);
  }
  for (int i_sf = 0; i_sf < 12; ++i_sf) {
    sum += score[m * 12 + i_sf];
  }
  for (int k_sf = 0; k_sf < 12; ++k_sf) {
    score[m * 12 + k_sf] = score[m * 12 + k_sf] / sum;
  }

  for (int n_fl = 0; n_fl < 512; ++n_fl) {
    output[i * 4096 * 12 * 512 + j * 12 * 512 + m * 512 + n_fl] = 0.0;
    for (int k_fl = 0; k_fl < 12; ++k_fl) {
      output[i * 4096 * 12 * 512 + j * 12 * 512 + m * 512 + n_fl] +=
          score[m * 12 + k_fl] *
          V[i * 4096 * 12 * 512 + j * 12 * 512 + k_fl * 512 + n_fl];
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

  mha<<<grid, block>>>(d_queries, d_keys, d_values, d_output);

  hipMemcpy(output, d_output, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_queries);
  hipFree(d_keys);
  hipFree(d_values);
  hipFree(d_output);
}
