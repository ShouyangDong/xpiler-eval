
__global__ void transpose(const float *__restrict__ input,
                          float *__restrict__ output) {
  int i0 = blockIdx.z;
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  const int d0 = 33;
  const int d1 = 40;
  const int d2 = 5;
  if (i0 < d0 && i1 < d1 && i2 < d2) {
    int in_idx = i0 * (d1 * d2) + i1 * d2 + i2;
    int out_idx = i0 * (d2 * d1) + i2 * d1 + i1;
    output[out_idx] = input[in_idx];
  }
}

extern "C" void transpose_kernel(float *input, float *output, int d0, int d1,
                                 int d2) {

  size_t total_elems = d0 * d1 * d2;
  size_t bytes = total_elems * sizeof(float);

  float *d_input, *d_output;
  cudaMalloc(&d_input, bytes);
  cudaMalloc(&d_output, bytes);

  cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((d2 + block.x - 1) / block.x, (d1 + block.y - 1) / block.y, d0);

  transpose<<<grid, block>>>(d_input, d_output);

  cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}
