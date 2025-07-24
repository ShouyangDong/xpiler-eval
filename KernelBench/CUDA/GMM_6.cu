
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ X, float* __restrict__ Y, float* __restrict__ Z);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ X, float* __restrict__ Y, float* __restrict__ Z) {
  float Z_local[32];
  __shared__ float X_shared[2048];
  __shared__ float Y_shared[2048];
  for (int i_4_init = 0; i_4_init < 2; ++i_4_init) {
    for (int vthread_x_s = 0; vthread_x_s < 16; ++vthread_x_s) {
      Z_local[((vthread_x_s * 2) + i_4_init)] = 0.000000e+00f;
    }
  }
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_ax2_fused_0 = 0; ax0_ax1_ax2_fused_0 < 16; ++ax0_ax1_ax2_fused_0) {
      X_shared[((ax0_ax1_ax2_fused_0 * 128) + ((int)threadIdx.x))] = X[((((((((int)blockIdx.x) >> 3) * 32768) + (ax0_ax1_ax2_fused_0 * 2048)) + ((((int)threadIdx.x) >> 5) * 512)) + (k_0 * 32)) + (((int)threadIdx.x) & 31))];
    }
    for (int ax0_ax1_ax2_fused_0_1 = 0; ax0_ax1_ax2_fused_0_1 < 4; ++ax0_ax1_ax2_fused_0_1) {
      *(float4*)(Y_shared + ((ax0_ax1_ax2_fused_0_1 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(Y + (((((((((int)blockIdx.x) >> 6) * 262144) + (k_0 * 16384)) + (ax0_ax1_ax2_fused_0_1 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    }
    __syncthreads();
    for (int k_2 = 0; k_2 < 32; ++k_2) {
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        for (int vthread_x_s_1 = 0; vthread_x_s_1 < 16; ++vthread_x_s_1) {
          Z_local[((vthread_x_s_1 * 2) + i_4)] = (Z_local[((vthread_x_s_1 * 2) + i_4)] + (X_shared[(((((vthread_x_s_1 >> 1) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_4 * 32)) + k_2)] * Y_shared[(((k_2 * 64) + ((vthread_x_s_1 & 1) * 32)) + (((int)threadIdx.x) & 31))]));
        }
      }
    }
  }
  for (int ax1 = 0; ax1 < 2; ++ax1) {
    for (int vthread_x_s_2 = 0; vthread_x_s_2 < 16; ++vthread_x_s_2) {
      Z[((((((((((int)blockIdx.x) >> 3) * 32768) + ((vthread_x_s_2 >> 1) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + (ax1 * 512)) + ((((int)blockIdx.x) & 7) * 64)) + ((vthread_x_s_2 & 1) * 32)) + (((int)threadIdx.x) & 31))] = Z_local[((vthread_x_s_2 * 2) + ax1)];
    }
  }
}

