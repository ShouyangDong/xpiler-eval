

__global__ void transpose(const float *__restrict__ input,
                          float *__restrict__ output) {
  const int d0 = 36;
  const int d1 = 16;
  const int d2 = 48;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < d0 && j < d1 && k < d2) {

    int in_idx = i * (d1 * d2) + j * d2 + k;
    int out_idx = i * (d2 * d1) + k * d1 + j;

    output[out_idx] = input[in_idx];
  }
}

extern "C" void transpose_kernel(float *host_input, float *host_output, int d0,
                                 int d1, int d2) {

  const int out_size = d0 * d2 * d1;
  const size_t in_bytes = (size_t)d0 * d1 * d2 * sizeof(float);
  const size_t out_bytes = (size_t)out_size * sizeof(float);

  float *d_input = nullptr;
  float *d_output = nullptr;

  hipMalloc(&d_input, in_bytes);
  hipMalloc(&d_output, out_bytes);

  hipMemcpy(d_input, host_input, in_bytes, hipMemcpyHostToDevice);

  dim3 block(8, 8, 8);
  dim3 grid((d0 + block.x - 1) / block.x, (d1 + block.y - 1) / block.y,
            (d2 + block.z - 1) / block.z);

  transpose<<<grid, block>>>(d_input, d_output);
  hipDeviceSynchronize();

  hipMemcpy(host_output, d_output, out_bytes, hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);
}