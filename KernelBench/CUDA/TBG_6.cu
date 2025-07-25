
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
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ C, float* __restrict__ query, float* __restrict__ value);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ C, float* __restrict__ query, float* __restrict__ value) {
  float C_local[8];
  __shared__ float query_T_shared[6144];
  __shared__ float value_T_shared[768];
  C_local[0] = 0.000000e+00f;
  C_local[1] = 0.000000e+00f;
  C_local[2] = 0.000000e+00f;
  C_local[3] = 0.000000e+00f;
  C_local[4] = 0.000000e+00f;
  C_local[5] = 0.000000e+00f;
  C_local[6] = 0.000000e+00f;
  C_local[7] = 0.000000e+00f;
  *(float4*)(query_T_shared + (((int)threadIdx.x) * 4)) = *(float4*)(query + (((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(query_T_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(query + ((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 49152));
  *(float4*)(query_T_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(query + ((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 98304));
  *(float4*)(query_T_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(query + ((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 147456));
  int2 v_ = make_int2(((((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 6) * 98304)) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 63) >> 2)))+(1536*0), ((((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 6) * 98304)) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 63) >> 2)))+(1536*1));
  *(float2*)(value_T_shared + (((int)threadIdx.x) * 2)) = make_float2(value[v_.x],value[v_.y]);
__asm__ __volatile__("cp.async.commit_group;");

  *(float4*)(query_T_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(query + ((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 16));
  *(float4*)(query_T_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(query + ((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 49168));
  *(float4*)(query_T_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(query + ((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 98320));
  *(float4*)(query_T_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(query + ((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 147472));
  int2 v__1 = make_int2((((((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 6) * 98304)) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 63) >> 2)) + 16))+(1536*0), (((((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 6) * 98304)) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + ((((int)threadIdx.x) & 63) >> 2)) + 16))+(1536*1));
  *(float2*)(value_T_shared + ((((int)threadIdx.x) * 2) + 256)) = make_float2(value[v__1.x],value[v__1.y]);
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0_fused = 0; k_0_fused < 6; ++k_0_fused) {
    __syncthreads();
    *(float4*)(query_T_shared + ((((k_0_fused + 2) % 3) * 2048) + (((int)threadIdx.x) * 4))) = *(float4*)(query + (((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + (k_0_fused * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 32));
    *(float4*)(query_T_shared + (((((k_0_fused + 2) % 3) * 2048) + (((int)threadIdx.x) * 4)) + 512)) = *(float4*)(query + (((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + (k_0_fused * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 49184));
    *(float4*)(query_T_shared + (((((k_0_fused + 2) % 3) * 2048) + (((int)threadIdx.x) * 4)) + 1024)) = *(float4*)(query + (((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + (k_0_fused * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 98336));
    *(float4*)(query_T_shared + (((((k_0_fused + 2) % 3) * 2048) + (((int)threadIdx.x) * 4)) + 1536)) = *(float4*)(query + (((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 2) * 1536)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + (k_0_fused * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 147488));
    int2 v__2 = make_int2(((((((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 6) * 98304)) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + (k_0_fused * 16)) + ((((int)threadIdx.x) & 63) >> 2)) + 32))+(1536*0), ((((((((((((int)blockIdx.x) / 96) * 196608) + ((((int)threadIdx.x) >> 6) * 98304)) + ((((int)blockIdx.x) & 7) * 12288)) + ((((int)threadIdx.x) & 3) * 3072)) + (((((int)blockIdx.x) % 96) >> 3) * 128)) + (k_0_fused * 16)) + ((((int)threadIdx.x) & 63) >> 2)) + 32))+(1536*1));
    *(float2*)(value_T_shared + ((((k_0_fused + 2) % 3) * 256) + (((int)threadIdx.x) * 2))) = make_float2(value[v__2.x],value[v__2.y]);
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

    __syncthreads();
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      C_local[0] = (C_local[0] + (query_T_shared[(((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1)] * value_T_shared[(((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3))]));
      C_local[1] = (C_local[1] + (query_T_shared[(((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1)] * value_T_shared[((((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
      C_local[2] = (C_local[2] + (query_T_shared[((((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1) + 256)] * value_T_shared[(((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3))]));
      C_local[3] = (C_local[3] + (query_T_shared[((((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1) + 256)] * value_T_shared[((((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
      C_local[4] = (C_local[4] + (query_T_shared[((((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1) + 512)] * value_T_shared[(((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3))]));
      C_local[5] = (C_local[5] + (query_T_shared[((((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1) + 512)] * value_T_shared[((((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
      C_local[6] = (C_local[6] + (query_T_shared[((((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1) + 768)] * value_T_shared[(((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3))]));
      C_local[7] = (C_local[7] + (query_T_shared[((((((k_0_fused % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1) + 768)] * value_T_shared[((((((k_0_fused % 3) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (k_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int k_1_1 = 0; k_1_1 < 16; ++k_1_1) {
    C_local[0] = (C_local[0] + (query_T_shared[((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1)] * value_T_shared[((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3))]));
    C_local[1] = (C_local[1] + (query_T_shared[((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
    C_local[2] = (C_local[2] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1) + 256)] * value_T_shared[((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3))]));
    C_local[3] = (C_local[3] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1) + 256)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
    C_local[4] = (C_local[4] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1) + 512)] * value_T_shared[((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3))]));
    C_local[5] = (C_local[5] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1) + 512)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
    C_local[6] = (C_local[6] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1) + 768)] * value_T_shared[((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3))]));
    C_local[7] = (C_local[7] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_1) + 768)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_1 * 8)) + (((int)threadIdx.x) & 3)) + 4)]));
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k_1_2 = 0; k_1_2 < 16; ++k_1_2) {
    C_local[0] = (C_local[0] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2048)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 256)]));
    C_local[1] = (C_local[1] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2048)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 260)]));
    C_local[2] = (C_local[2] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2304)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 256)]));
    C_local[3] = (C_local[3] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2304)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 260)]));
    C_local[4] = (C_local[4] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2560)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 256)]));
    C_local[5] = (C_local[5] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2560)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 260)]));
    C_local[6] = (C_local[6] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2816)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 256)]));
    C_local[7] = (C_local[7] + (query_T_shared[(((((((int)threadIdx.x) >> 6) * 1024) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_1_2) + 2816)] * value_T_shared[(((((((int)threadIdx.x) >> 6) * 128) + (k_1_2 * 8)) + (((int)threadIdx.x) & 3)) + 260)]));
  }
  C[(((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3))] = C_local[0];
  C[((((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = C_local[1];
  C[((((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3)) + 1024)] = C_local[2];
  C[((((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3)) + 1028)] = C_local[3];
  C[((((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3)) + 2048)] = C_local[4];
  C[((((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3)) + 2052)] = C_local[5];
  C[((((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3)) + 3072)] = C_local[6];
  C[((((((((((int)blockIdx.x) / 96) * 98304) + ((((int)threadIdx.x) >> 6) * 49152)) + (((((int)blockIdx.x) % 96) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 3)) + 3076)] = C_local[7];
}

