
constexpr int D0 = 1000;
constexpr int D1 = 2048;
constexpr int D2 = 8;
constexpr int TOTAL_PARAMS = D0 * D1 * D2;

__global__ void gather(const float *params, const int64_t *indices,
                       float *output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = D0 * D1 * N;
  if (tid >= total_elements)
    return;

  int i = tid / (D1 * N);
  int rem = tid % (D1 * N);
  int j = rem / N;
  int n = rem % N;

  int64_t src_idx = indices[n];

  float val = 0.0f;
  if (src_idx >= 0 && src_idx < D2) {
    val = params[i * D1 * D2 + j * D2 + src_idx];
  }

  output[tid] = val;
}

extern "C" void gather_kernel(const float *h_params, const int64_t *h_indices,
                              float *h_output, int N) {

  size_t params_bytes = D0 * D1 * D2 * sizeof(float);
  size_t indices_bytes = N * sizeof(int64_t);
  size_t output_bytes = D0 * D1 * N * sizeof(float);

  float *d_params;
  int64_t *d_indices;
  float *d_output;

  cudaMalloc(&d_params, params_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  const int block_size = 1024;
  int total_threads = D0 * D1 * N;
  int grid_size = (total_threads + block_size - 1) / block_size;

  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output, N);

  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_params);
  cudaFree(d_indices);
  cudaFree(d_output);
}
