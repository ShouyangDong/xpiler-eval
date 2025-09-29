

constexpr int N = 1;
constexpr int C = 64;
constexpr int H = 112;
constexpr int W = 56;
constexpr int TOTAL_ELEMENTS = N * C * H * W;
constexpr int OUTPUT_H =
    H * 2; // Concatenating two tensors along height axis -> 224 height
constexpr int OUTPUT_TOTAL_ELEMENTS = N * C * OUTPUT_H * W;

__global__ void concat(const float *__restrict__ input1,
                       const float *__restrict__ input2,
                       float *__restrict__ output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= OUTPUT_TOTAL_ELEMENTS)
    return;

  // Decode output index
  int n = tid / (C * OUTPUT_H * W);
  int rem = tid % (C * OUTPUT_H * W);
  int c = rem / (OUTPUT_H * W);
  rem = rem % (OUTPUT_H * W);
  int h = rem / W;
  int w = rem % W;

  if (h < H) {
    // First half (original height) comes from input1
    output[tid] = input1[n * C * H * W + c * H * W + h * W + w];
  } else {
    // Second half comes from input2
    int h2 = h - H;
    output[tid] = input2[n * C * H * W + c * H * W + h2 * W + w];
  }
}

extern "C" void concat_kernel(const float *h_input1, const float *h_input2,
                              float *h_output) {
  size_t input_bytes = TOTAL_ELEMENTS * sizeof(float);
  size_t output_bytes = OUTPUT_TOTAL_ELEMENTS * sizeof(float);

  float *d_input1;
  float *d_input2;
  float *d_output;

  cudaMalloc(&d_input1, input_bytes);
  cudaMalloc(&d_input2, input_bytes);
  cudaMalloc(&d_output, output_bytes);

  cudaMemcpy(d_input1, h_input1, input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input2, h_input2, input_bytes, cudaMemcpyHostToDevice);

  const int block_size = 256;
  int total_threads = OUTPUT_TOTAL_ELEMENTS;
  int grid_size = (total_threads + block_size - 1) / block_size;

  concat<<<grid_size, block_size>>>(d_input1, d_input2, d_output);

  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input1);
  cudaFree(d_input2);
  cudaFree(d_output);
}
