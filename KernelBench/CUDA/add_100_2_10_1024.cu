__global__ void __launch_bounds__(1024)
    add(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0;
       ax0_ax1_fused_ax2_fused_ax3_fused_outer < 8;
       ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
         ((int)blockIdx.x)) < 2048000) {
      T_add[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
              (((int)blockIdx.x) * 1024)) +
             ((int)threadIdx.x))] =
          (A[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
               (((int)blockIdx.x) * 1024)) +
              ((int)threadIdx.x))] +
           B[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
               (((int)blockIdx.x) * 1024)) +
              ((int)threadIdx.x))]);
    }
  }
}

extern "C" void add_kernel(float *A, float *B, float *C, int size) {
  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_B, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks(256);
  add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
