__global__ void __launch_bounds__(1024)
    maxpool(float *__restrict__ A, float *__restrict__ pool_max) {
  float pool_max_local[1];
  pool_max_local[0] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_max_local[0] =
          max(pool_max_local[0],
              A[(((((((((int)threadIdx.x) >> 8) * 4096) +
                     (((((int)threadIdx.x) & 255) >> 7) * 1536)) +
                    (rv0 * 512)) +
                   (((((int)threadIdx.x) & 127) >> 6) * 192)) +
                  (rv1 * 64)) +
                 (((int)threadIdx.x) & 63))]);
    }
  }
  pool_max[((int)threadIdx.x)] = pool_max_local[0];
}

extern "C" void maxpool_kernel(float *input, float *output, int batch_size,
                               int channels, int input_H, int kernel_size,
                               int stride) {
  float *d_input;
  float *d_output;
  int output_H = (input_H - kernel_size) / stride + 1;
  int input_size = batch_size * input_H * input_H * channels;
  int output_size = batch_size * output_H * output_H * channels;
  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((output_size + blockSize.x - 1) / blockSize.x);

  maxpool<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}
