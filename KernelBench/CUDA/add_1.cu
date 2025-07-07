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
