__global__ void __launch_bounds__(1024)
    add(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0;
       ax0_ax1_fused_ax2_fused_ax3_fused_outer < 143;
       ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 14336) +
         ((int)blockIdx.x)) < 2048000) {
      T_add[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 14336) +
              (((int)blockIdx.x) * 1024)) +
             ((int)threadIdx.x))] =
          (A[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 14336) +
               (((int)blockIdx.x) * 1024)) +
              ((int)threadIdx.x))] +
           B[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 14336) +
               (((int)blockIdx.x) * 1024)) +
              ((int)threadIdx.x))]);
    }
  }
}

extern "C" void add_kernel(float *A, float *B, float *C, int size) {
  float *d_A;
  float *d_B;
  float *d_C;

  hipMalloc(&d_A, size * sizeof(float));
  hipMalloc(&d_B, size * sizeof(float));
  hipMalloc(&d_C, size * sizeof(float));

  hipMemcpy(d_A, A, size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, size * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks(14);
  add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
