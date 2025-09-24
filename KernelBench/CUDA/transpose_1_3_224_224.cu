__global__ void transpose(const float *__restrict__ input,
                          float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = 1;
  const int C = 3;
  const int H = 224;
  const int W = 224;
  int total = N * C * H * W;

  if (idx < total) {

    int n = idx / (C * H * W);
    int c = (idx / (H * W)) % C;
    int h = (idx / W) % H;
    int out_idx = n * (H * W * C) + ((idx / W) % H) * (W * C) + (idx % W) * C +
                  ((idx / (H * W)) % C);

    int in_idx = n * (C * H * W) + ((idx / (H * W)) % C) * (H * W) +
                 ((idx / W) % H) * W + (idx % W);

    output[out_idx] = input[in_idx];
  }
}

extern "C" void transpose_kernel(float *input, float *output, int N, int C,
                                 int H, int W) {
  int total = N * C * H * W;

  float *d_input, *d_output;
  cudaMalloc(&d_input, total * sizeof(float));
  cudaMalloc(&d_output, total * sizeof(float));

  cudaMemcpy(d_input, input, total * sizeof(float), cudaMemcpyHostToDevice);
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  transpose<<<blocks, threads>>>(d_input, d_output);

  cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaDeviceSynchronize();
}
