

constexpr int N = 1;
constexpr int C = 512;
constexpr int H = 7;
constexpr int W = 7;
constexpr int TOTAL_ELEMENTS = N * C * H * W;

__global__ void scatter(const float *__restrict__ input,
                               const int *__restrict__ indices,
                               float *__restrict__ output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= TOTAL_ELEMENTS) return;

  // Decode thread index
  int n = tid / (C * H * W);
  int rem = tid % (C * H * W);
  int c_idx = rem / (H * W);
  rem = rem % (H * W);
  int h = rem / W;
  int w = rem % W;

  // Get target index from indices tensor for axis=1 (channel)
  int target_c = indices[tid];
  
  // Bounds check
  if (target_c >= 0 && target_c < 512) {
    // Calculate output index: scatter input[n][c_idx][h][w] -> output[n][target_c][h][w]
    int output_idx = n * 512 * H * W + target_c * H * W + h * W + w;
    output[output_idx] = input[tid];
  }
}

extern "C" void scatter_kernel(const float *h_input, const int *h_indices,
                              float *h_output) {
  size_t input_bytes = TOTAL_ELEMENTS * sizeof(float);
  size_t indices_bytes = TOTAL_ELEMENTS * sizeof(int);
  size_t output_bytes = N * 512 * H * W * sizeof(float); // output has shape [1,512,7,7]

  float *d_input;
  int *d_indices;
  float *d_output;

  cudaMalloc(&d_input, input_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  // Copy input data (acts as base for output)
  cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  // Initialize output with input values
  cudaMemcpy(d_output, d_input, input_bytes, cudaMemcpyDeviceToDevice);

  const int block_size = 256;
  int total_threads = TOTAL_ELEMENTS;
  int grid_size = (total_threads + block_size - 1) / block_size;

  scatter<<<grid_size, block_size>>>(d_input, d_indices, d_output);

  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_indices);
  cudaFree(d_output);
}