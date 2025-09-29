

constexpr int N = 1;
constexpr int C = 64;
constexpr int H = 56;
constexpr int W = 112;
constexpr int TOTAL_ELEMENTS = N * C * H * W;
constexpr int OUTPUT_W =
    W * 2; // Concatenating two tensors along width axis -> 224 width
constexpr int OUTPUT_TOTAL_ELEMENTS = N * C * H * OUTPUT_W;

__global__ void concat(const float *__restrict__ input1,
                       const float *__restrict__ input2,
                       float *__restrict__ output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= OUTPUT_TOTAL_ELEMENTS)
    return;

  // Decode output index
  int n = tid / (C * H * OUTPUT_W);
  int rem = tid % (C * H * OUTPUT_W);
  int c = rem / (H * OUTPUT_W);
  rem = rem % (H * OUTPUT_W);
  int h = rem / OUTPUT_W;
  int w = rem % OUTPUT_W;

  if (w < W) {
    // First half (original width) comes from input1
    output[tid] = input1[n * C * H * W + c * H * W + h * W + w];
  } else {
    // Second half comes from input2
    int w2 = w - W;
    output[tid] = input2[n * C * H * W + c * H * W + h * W + w2];
  }
}

extern "C" void concat_kernel(const float *h_input1, const float *h_input2,
                              float *h_output) {
  size_t input_bytes = TOTAL_ELEMENTS * sizeof(float);
  size_t output_bytes = OUTPUT_TOTAL_ELEMENTS * sizeof(float);

  float *d_input1;
  float *d_input2;
  float *d_output;

  hipMalloc(&d_input1, input_bytes);
  hipMalloc(&d_input2, input_bytes);
  hipMalloc(&d_output, output_bytes);

  hipMemcpy(d_input1, h_input1, input_bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_input2, h_input2, input_bytes, hipMemcpyHostToDevice);

  const int block_size = 256;
  int total_threads = OUTPUT_TOTAL_ELEMENTS;
  int grid_size = (total_threads + block_size - 1) / block_size;

  concat<<<grid_size, block_size>>>(d_input1, d_input2, d_output);

  hipMemcpy(h_output, d_output, output_bytes, hipMemcpyDeviceToHost);

  hipFree(d_input1);
  hipFree(d_input2);
  hipFree(d_output);
}