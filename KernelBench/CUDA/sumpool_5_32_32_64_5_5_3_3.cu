__global__ void __launch_bounds__(1024)
    sumpool(float *__restrict__ A, float *__restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 125) {
        pool_sum[0] =
            (pool_sum[0] +
             A[(((((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) /
                      25) *
                     65536) +
                    (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) %
                       50) /
                      5) *
                     6144)) +
                   (rv0 * 2048)) +
                  ((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 6)) %
                    10) *
                   192)) +
                 (rv1 * 64)) +
                (((int)threadIdx.x) & 63))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 125) {
    pool_avg[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_sum[0];
  }
}

extern "C" void sumpool_kernel(float *input, float *output, int batch_size,
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

  sumpool<<<numBlocks, blockSize>>>(d_input, d_output);

  cudaMemcpy(output, d_output, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}
