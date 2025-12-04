

constexpr int N = 1;
constexpr int C = 64;
constexpr int H = 56;
constexpr int W = 56;
constexpr int TOTAL_ELEMENTS = N * C * H * W;

__global__ void scatter(const float *__restrict__ input,
                        const int *__restrict__ indices,
                        float *__restrict__ output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= TOTAL_ELEMENTS)
    return;

  int n = tid / (C * H * W);
  int rem = tid % (C * H * W);
  int c_idx = rem / (H * W);
  rem = rem % (H * W);
  int h = rem / W;
  int w = rem % W;

  int target_c = indices[tid];

  if (target_c >= 0 && target_c < 64) {

    int output_idx = n * 64 * H * W + target_c * H * W + h * W + w;
    output[output_idx] = input[tid];
  }
}

extern "C" void scatter_kernel(const float *h_input, const int *h_indices,
                               float *h_output) {
  size_t input_bytes = TOTAL_ELEMENTS * sizeof(float);
  size_t indices_bytes = TOTAL_ELEMENTS * sizeof(int);
  size_t output_bytes = N * 64 * H * W * sizeof(float);

  float *d_input;
  int *d_indices;
  float *d_output;

  cudaMalloc(&d_input, input_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  cudaMemcpy(d_output, d_input, input_bytes, cudaMemcpyDeviceToDevice);

  const int block_size = 1024;
  int total_threads = TOTAL_ELEMENTS;
  int grid_size = (total_threads + block_size - 1) / block_size;

  scatter<<<grid_size, block_size>>>(d_input, d_indices, d_output);

  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_indices);
  cudaFree(d_output);
}