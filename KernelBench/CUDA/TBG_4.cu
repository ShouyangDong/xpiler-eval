
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ C, float* __restrict__ query, float* __restrict__ value);
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ C, float* __restrict__ query, float* __restrict__ value) {
  float C_local[16];
  __shared__ float query_T_shared[256];
  __shared__ float value_T_shared[128];
  for (int i_3_init = 0; i_3_init < 2; ++i_3_init) {
    for (int j_4_init = 0; j_4_init < 2; ++j_4_init) {
      C_local[((i_3_init * 2) + j_4_init)] = 0.000000e+00f;
      C_local[(((i_3_init * 2) + j_4_init) + 4)] = 0.000000e+00f;
      C_local[(((i_3_init * 2) + j_4_init) + 8)] = 0.000000e+00f;
      C_local[(((i_3_init * 2) + j_4_init) + 12)] = 0.000000e+00f;
    }
  }
  for (int k_0 = 0; k_0 < 8; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 4; ++ax0_ax1_ax2_ax3_fused_0) {
      *(float2*)(query_T_shared + ((ax0_ax1_ax2_ax3_fused_0 * 64) + (((int)threadIdx.x) * 2))) = *(float2*)(query + ((((((((((int)blockIdx.x) / 384) * 98304) + (((((int)blockIdx.x) & 31) >> 3) * 24576)) + (ax0_ax1_ax2_ax3_fused_0 * 6144)) + ((((int)threadIdx.x) >> 2) * 768)) + (((((int)blockIdx.x) % 384) >> 5) * 64)) + (k_0 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    int4 v_ = make_int4(((((((((((int)blockIdx.x) / 384) * 98304) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 384) >> 5) * 64)) + (k_0 * 8)) + (((int)threadIdx.x) >> 2)))+(768*0), ((((((((((int)blockIdx.x) / 384) * 98304) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 384) >> 5) * 64)) + (k_0 * 8)) + (((int)threadIdx.x) >> 2)))+(768*1), ((((((((((int)blockIdx.x) / 384) * 98304) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 384) >> 5) * 64)) + (k_0 * 8)) + (((int)threadIdx.x) >> 2)))+(768*2), ((((((((((int)blockIdx.x) / 384) * 98304) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 384) >> 5) * 64)) + (k_0 * 8)) + (((int)threadIdx.x) >> 2)))+(768*3));
    *(float4*)(value_T_shared + (((int)threadIdx.x) * 4)) = make_float4(value[v_.x],value[v_.y],value[v_.z],value[v_.w]);
    __syncthreads();
    for (int k_1 = 0; k_1 < 8; ++k_1) {
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        for (int j_4 = 0; j_4 < 2; ++j_4) {
          C_local[((i_3 * 2) + j_4)] = (C_local[((i_3 * 2) + j_4)] + (query_T_shared[((((((int)threadIdx.x) >> 3) * 16) + (i_3 * 8)) + k_1)] * value_T_shared[(((k_1 * 16) + ((((int)threadIdx.x) & 7) * 2)) + j_4)]));
          C_local[(((i_3 * 2) + j_4) + 4)] = (C_local[(((i_3 * 2) + j_4) + 4)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (i_3 * 8)) + k_1) + 64)] * value_T_shared[(((k_1 * 16) + ((((int)threadIdx.x) & 7) * 2)) + j_4)]));
          C_local[(((i_3 * 2) + j_4) + 8)] = (C_local[(((i_3 * 2) + j_4) + 8)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (i_3 * 8)) + k_1) + 128)] * value_T_shared[(((k_1 * 16) + ((((int)threadIdx.x) & 7) * 2)) + j_4)]));
          C_local[(((i_3 * 2) + j_4) + 12)] = (C_local[(((i_3 * 2) + j_4) + 12)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (i_3 * 8)) + k_1) + 192)] * value_T_shared[(((k_1 * 16) + ((((int)threadIdx.x) & 7) * 2)) + j_4)]));
        }
      }
    }
  }
  for (int ax2 = 0; ax2 < 2; ++ax2) {
    for (int ax3 = 0; ax3 < 2; ++ax3) {
      C[(((((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + (ax2 * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3)] = C_local[((ax2 * 2) + ax3)];
      C[((((((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + (ax2 * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3) + 1024)] = C_local[(((ax2 * 2) + ax3) + 4)];
      C[((((((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + (ax2 * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3) + 2048)] = C_local[(((ax2 * 2) + ax3) + 8)];
      C[((((((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + (ax2 * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3) + 3072)] = C_local[(((ax2 * 2) + ax3) + 12)];
    }
  }
}

