__global__ void transpose(const float *__restrict__ input,
                          float *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = 1;
  const int C = 3;
  const int H = 224;
  const int W = 224;
  int total = N * C * H * W;

  if (idx < total) {
    int w = idx % W; 
    int h = idx % H; 
    int c = idx % C; 
    int n = idx; 
    int out_idx = ((n * W + w) * C + c) * H + h; 
    output[out_idx] = input[idx];
  }
}

extern "C" void transpose_kernel(float *input, float *output, int N, int C,
                                 int H, int W) {
  int total = N * C * H * W;

  float *d_input, *d_output;
  cudaMalloc(&d_input, total * sizeof(float));
  cudaMalloc(&d_output, total * sizeof(float));

  cudaMemcpy(d_input, input, total * sizeof(float), cudaMemcpyHostToDevice);
  int threads = 1024;
  int blocks = (total + threads - 1) / threads;

  transpose<<<blocks, threads>>>(d_input, d_output);

  cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaDeviceSynchronize();
}
