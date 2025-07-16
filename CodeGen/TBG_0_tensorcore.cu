
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
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ C, float* __restrict__ query, float* __restrict__ value);
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ C, float* __restrict__ query, float* __restrict__ value) {
  float C_local[8];
  __shared__ float query_T_shared[6144];
  __shared__ float value_T_shared[768];
  for (int j_3_init = 0; j_3_init < 2; ++j_3_init) {
    C_local[j_3_init] = 0.000000e+00f;
    C_local[(j_3_init + 2)] = 0.000000e+00f;
    C_local[(j_3_init + 4)] = 0.000000e+00f;
    C_local[(j_3_init + 6)] = 0.000000e+00f;
  }
  for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 2; ++ax0_ax1_ax2_ax3_fused_0) {
    *(float4*)(query_T_shared + ((ax0_ax1_ax2_ax3_fused_0 * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(query + ((((ax0_ax1_ax2_ax3_fused_0 * 49152) + ((((int)threadIdx.x) >> 2) * 768)) + ((((int)blockIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 3) * 4)));
  }
  value_T_shared[((int)threadIdx.x)] = value[(((((((int)blockIdx.x) & 7) * 12288) + ((((int)threadIdx.x) & 15) * 768)) + ((((int)blockIdx.x) >> 3) * 64)) + (((int)threadIdx.x) >> 4))];
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax0_ax1_ax2_ax3_fused_0_1 = 0; ax0_ax1_ax2_ax3_fused_0_1 < 2; ++ax0_ax1_ax2_ax3_fused_0_1) {
    *(float4*)(query_T_shared + (((ax0_ax1_ax2_ax3_fused_0_1 * 1024) + (((int)threadIdx.x) * 4)) + 2048)) = *(float4*)(query + (((((ax0_ax1_ax2_ax3_fused_0_1 * 49152) + ((((int)threadIdx.x) >> 2) * 768)) + ((((int)blockIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 3) * 4)) + 16));
  }
  value_T_shared[(((int)threadIdx.x) + 256)] = value[((((((((int)blockIdx.x) & 7) * 12288) + ((((int)threadIdx.x) & 15) * 768)) + ((((int)blockIdx.x) >> 3) * 64)) + (((int)threadIdx.x) >> 4)) + 16)];
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0_fused = 0; k_0_fused < 2; ++k_0_fused) {
    __syncthreads();
    for (int ax0_ax1_ax2_ax3_fused_0_2 = 0; ax0_ax1_ax2_ax3_fused_0_2 < 2; ++ax0_ax1_ax2_ax3_fused_0_2) {
      *(float4*)(query_T_shared + (((((k_0_fused + 2) % 3) * 2048) + (ax0_ax1_ax2_ax3_fused_0_2 * 1024)) + (((int)threadIdx.x) * 4))) = *(float4*)(query + ((((((ax0_ax1_ax2_ax3_fused_0_2 * 49152) + ((((int)threadIdx.x) >> 2) * 768)) + ((((int)blockIdx.x) >> 3) * 64)) + (k_0_fused * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 32));
    }
    value_T_shared[((((k_0_fused + 2) % 3) * 256) + ((int)threadIdx.x))] = value[(((((((((int)blockIdx.x) & 7) * 12288) + ((((int)threadIdx.x) & 15) * 768)) + ((((int)blockIdx.x) >> 3) * 64)) + (k_0_fused * 16)) + (((int)threadIdx.x) >> 4)) + 32)];
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

    __syncthreads();
    for (int k_1 = 0; k_1 < 4; ++k_1) {
      for (int j_3 = 0; j_3 < 2; ++j_3) {
        for (int k_2 = 0; k_2 < 4; ++k_2) {
          C_local[j_3] = (C_local[j_3] + (query_T_shared[((((k_0_fused * 2048) + ((((int)threadIdx.x) >> 3) * 16)) + (k_1 * 4)) + k_2)] * value_T_shared[(((((k_0_fused * 256) + (k_1 * 64)) + (k_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3)]));
          C_local[(j_3 + 2)] = (C_local[(j_3 + 2)] + (query_T_shared[(((((k_0_fused * 2048) + ((((int)threadIdx.x) >> 3) * 16)) + (k_1 * 4)) + k_2) + 512)] * value_T_shared[(((((k_0_fused * 256) + (k_1 * 64)) + (k_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3)]));
          C_local[(j_3 + 4)] = (C_local[(j_3 + 4)] + (query_T_shared[(((((k_0_fused * 2048) + ((((int)threadIdx.x) >> 3) * 16)) + (k_1 * 4)) + k_2) + 1024)] * value_T_shared[(((((k_0_fused * 256) + (k_1 * 64)) + (k_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3)]));
          C_local[(j_3 + 6)] = (C_local[(j_3 + 6)] + (query_T_shared[(((((k_0_fused * 2048) + ((((int)threadIdx.x) >> 3) * 16)) + (k_1 * 4)) + k_2) + 1536)] * value_T_shared[(((((k_0_fused * 256) + (k_1 * 64)) + (k_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3)]));
        }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int k_1_1 = 0; k_1_1 < 4; ++k_1_1) {
    for (int j_3_1 = 0; j_3_1 < 2; ++j_3_1) {
      for (int k_2_1 = 0; k_2_1 < 4; ++k_2_1) {
        C_local[j_3_1] = (C_local[j_3_1] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (k_1_1 * 4)) + k_2_1) + 4096)] * value_T_shared[(((((k_1_1 * 64) + (k_2_1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_1) + 512)]));
        C_local[(j_3_1 + 2)] = (C_local[(j_3_1 + 2)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (k_1_1 * 4)) + k_2_1) + 4608)] * value_T_shared[(((((k_1_1 * 64) + (k_2_1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_1) + 512)]));
        C_local[(j_3_1 + 4)] = (C_local[(j_3_1 + 4)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (k_1_1 * 4)) + k_2_1) + 5120)] * value_T_shared[(((((k_1_1 * 64) + (k_2_1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_1) + 512)]));
        C_local[(j_3_1 + 6)] = (C_local[(j_3_1 + 6)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (k_1_1 * 4)) + k_2_1) + 5632)] * value_T_shared[(((((k_1_1 * 64) + (k_2_1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_1) + 512)]));
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k_1_2 = 0; k_1_2 < 4; ++k_1_2) {
    for (int j_3_2 = 0; j_3_2 < 2; ++j_3_2) {
      for (int k_2_2 = 0; k_2_2 < 4; ++k_2_2) {
        C_local[j_3_2] = (C_local[j_3_2] + (query_T_shared[((((((int)threadIdx.x) >> 3) * 16) + (k_1_2 * 4)) + k_2_2)] * value_T_shared[((((k_1_2 * 64) + (k_2_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_2)]));
        C_local[(j_3_2 + 2)] = (C_local[(j_3_2 + 2)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (k_1_2 * 4)) + k_2_2) + 512)] * value_T_shared[((((k_1_2 * 64) + (k_2_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_2)]));
        C_local[(j_3_2 + 4)] = (C_local[(j_3_2 + 4)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (k_1_2 * 4)) + k_2_2) + 1024)] * value_T_shared[((((k_1_2 * 64) + (k_2_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_2)]));
        C_local[(j_3_2 + 6)] = (C_local[(j_3_2 + 6)] + (query_T_shared[(((((((int)threadIdx.x) >> 3) * 16) + (k_1_2 * 4)) + k_2_2) + 1536)] * value_T_shared[((((k_1_2 * 64) + (k_2_2 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + j_3_2)]));
      }
    }
  }
  for (int ax3 = 0; ax3 < 2; ++ax3) {
    C[((((((((int)blockIdx.x) >> 3) * 16384) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3)] = C_local[ax3];
    C[(((((((((int)blockIdx.x) >> 3) * 16384) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3) + 4096)] = C_local[(ax3 + 2)];
    C[(((((((((int)blockIdx.x) >> 3) * 16384) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3) + 8192)] = C_local[(ax3 + 4)];
    C[(((((((((int)blockIdx.x) >> 3) * 16384) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax3) + 12288)] = C_local[(ax3 + 6)];
  }
}

