__global__ void transpose(const float *__restrict__ input,
                          float *__restrict__ output) {
  const int N = 1;
  const int C = 3;
  const int H = 224;
  const int W = 224;

  int w = blockIdx.x * 16 + threadIdx.x;
  int h = blockIdx.y * 16 + threadIdx.y;
  int nc = blockIdx.z;
  int n = nc / C;
  int c = nc % C;

  if (n < N && c < C && h < H && w < W) {
    int in_idx = ((n * C + c) * H + h) * W + w;
    int out_idx = ((n * W + w) * C + c) * H + h;
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
  dim3 block(16, 16);
  dim3 grid((W + 16 - 1) / 16, (H + 16 - 1) / 16, N * C);

  transpose<<<grid, block>>>(input, output);

  hipMemcpy(output, d_output, total * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}
