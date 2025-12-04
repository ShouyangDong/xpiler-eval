__global__ void transpose(const float *__restrict__ input,
                          float *__restrict__ output) {

  const int N = 2;
  const int C = 32;
  const int H = 4;
  const int W = 64;
  int w = blockIdx.x * 16 + threadIdx.x;
  int h = blockIdx.y * 4 + threadIdx.y;
  int n = blockIdx.z / C;
  int c = blockIdx.z % C;

  if (n < N && c < C && h < H && w < W) {
    int in_idx = ((n * C + c) * H + h) * W + w;
    int out_idx = ((n * H + h) * W + w) * C + c;
    output[out_idx] = input[in_idx];
  }
}

extern "C" void transpose_kernel(float *input, float *output, int N, int C,
                                 int H, int W) {

  int total = N * C * H * W;
  float *d_input, *d_output;
  hipMalloc(&d_input, total * sizeof(float));
  hipMalloc(&d_output, total * sizeof(float));

  hipMemcpy(d_input, input, total * sizeof(float), hipMemcpyHostToDevice);
  dim3 block(16, 4);
  dim3 grid((W + 16 - 1) / 16, (H + 4 - 1) / 4, N * C);
  transpose<<<grid, block>>>(d_input, d_output);

  hipMemcpy(output, d_output, total * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}
