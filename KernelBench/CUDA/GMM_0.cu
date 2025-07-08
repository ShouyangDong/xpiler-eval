
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
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ X, float* __restrict__ Y, float* __restrict__ Z);
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ X, float* __restrict__ Y, float* __restrict__ Z) {
  float Z_local[4];
  __shared__ float X_shared[512];
  __shared__ float Y_shared[8192];
  Z_local[0] = 0.000000e+00f;
  Z_local[1] = 0.000000e+00f;
  Z_local[2] = 0.000000e+00f;
  Z_local[3] = 0.000000e+00f;
  *(float4*)(X_shared + (((int)threadIdx.x) * 4)) = *(float4*)(X + (((((int)blockIdx.x) >> 1) * 512) + (((int)threadIdx.x) * 4)));
  *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(X + ((((((int)blockIdx.x) >> 1) * 512) + (((int)threadIdx.x) * 4)) + 256));
  *(float4*)(Y_shared + (((int)threadIdx.x) * 4)) = *(float4*)(Y + ((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 512));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1024));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1536));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2048));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2560));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3072));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3584));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4096));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4608));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5120));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5632));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6144));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6656));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7168));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7680));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 8192));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 4352)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 8704));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 4608)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 9216));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 4864)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 9728));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 5120)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 10240));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 5376)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 10752));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 5632)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 11264));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 5888)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 11776));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 6144)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 12288));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 6400)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 12800));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 6656)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 13312));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 6912)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 13824));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 7168)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 14336));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 7424)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 14848));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 7680)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 15360));
  *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 7936)) = *(float4*)(Y + (((((((int)threadIdx.x) >> 4) * 128) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 15872));
  __syncthreads();
  for (int k_1 = 0; k_1 < 64; ++k_1) {
    Z_local[0] = (Z_local[0] + (X_shared[(k_1 * 2)] * Y_shared[((k_1 * 128) + ((int)threadIdx.x))]));
    Z_local[1] = (Z_local[1] + (X_shared[((k_1 * 2) + 128)] * Y_shared[((k_1 * 128) + ((int)threadIdx.x))]));
    Z_local[2] = (Z_local[2] + (X_shared[((k_1 * 2) + 256)] * Y_shared[((k_1 * 128) + ((int)threadIdx.x))]));
    Z_local[3] = (Z_local[3] + (X_shared[((k_1 * 2) + 384)] * Y_shared[((k_1 * 128) + ((int)threadIdx.x))]));
    Z_local[0] = (Z_local[0] + (X_shared[((k_1 * 2) + 1)] * Y_shared[(((k_1 * 128) + ((int)threadIdx.x)) + 64)]));
    Z_local[1] = (Z_local[1] + (X_shared[((k_1 * 2) + 129)] * Y_shared[(((k_1 * 128) + ((int)threadIdx.x)) + 64)]));
    Z_local[2] = (Z_local[2] + (X_shared[((k_1 * 2) + 257)] * Y_shared[(((k_1 * 128) + ((int)threadIdx.x)) + 64)]));
    Z_local[3] = (Z_local[3] + (X_shared[((k_1 * 2) + 385)] * Y_shared[(((k_1 * 128) + ((int)threadIdx.x)) + 64)]));
  }
  Z[((((((int)blockIdx.x) >> 1) * 512) + ((((int)blockIdx.x) & 1) * 64)) + ((int)threadIdx.x))] = Z_local[0];
  Z[(((((((int)blockIdx.x) >> 1) * 512) + ((((int)blockIdx.x) & 1) * 64)) + ((int)threadIdx.x)) + 128)] = Z_local[1];
  Z[(((((((int)blockIdx.x) >> 1) * 512) + ((((int)blockIdx.x) & 1) * 64)) + ((int)threadIdx.x)) + 256)] = Z_local[2];
  Z[(((((((int)blockIdx.x) >> 1) * 512) + ((((int)blockIdx.x) & 1) * 64)) + ((int)threadIdx.x)) + 384)] = Z_local[3];
}

