
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
  __shared__ float X_shared[4096];
  __shared__ float Y_shared[4096];
  Z_local[0] = 0.000000e+00f;
  Z_local[2] = 0.000000e+00f;
  Z_local[1] = 0.000000e+00f;
  Z_local[3] = 0.000000e+00f;
  for (int k_0 = 0; k_0 < 2; ++k_0) {
    __syncthreads();
    *(float4*)(X_shared + (((int)threadIdx.x) * 4)) = *(float4*)(X + ((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 512));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 1024));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 1536));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 2048));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 2560));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 3072));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 3584));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 4096));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 4608));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 5120));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 5632));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 6144));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 6656));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 7168));
    *(float4*)(X_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(X + (((((((int)blockIdx.x) >> 1) * 8192) + (k_0 * 256)) + (((int)threadIdx.x) * 4)) + 7680));
    *(float4*)(Y_shared + (((int)threadIdx.x) * 4)) = *(float4*)(Y + ((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 512));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1024));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1536));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2048));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2560));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 3072));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 3584));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 4096));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 4608));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 5120));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 5632));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 6144));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 6656));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 7168));
    *(float4*)(Y_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(Y + (((((k_0 * 8192) + ((((int)threadIdx.x) >> 2) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 7680));
    __syncthreads();
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      Z_local[0] = (Z_local[0] + (X_shared[(((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16))] * Y_shared[((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2))]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2048)] * Y_shared[((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2))]));
      Z_local[1] = (Z_local[1] + (X_shared[(((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16))] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 1)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2048)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 1)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 1)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 16)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2049)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 16)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 1)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 17)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2049)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 17)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 32)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2050)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 32)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 33)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2050)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 33)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 3)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 48)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2051)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 48)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 3)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 49)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2051)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 49)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 4)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 64)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2052)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 64)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 4)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 65)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2052)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 65)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 5)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 80)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2053)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 80)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 5)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 81)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2053)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 81)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 6)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 96)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2054)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 96)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 6)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 97)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2054)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 97)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 7)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 112)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2055)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 112)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 7)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 113)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2055)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 113)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 8)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 128)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2056)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 128)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 8)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 129)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2056)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 129)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 9)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 144)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2057)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 144)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 9)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 145)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2057)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 145)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 10)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 160)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2058)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 160)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 10)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 161)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2058)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 161)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 11)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 176)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2059)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 176)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 11)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 177)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2059)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 177)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 12)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 192)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2060)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 192)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 12)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 193)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2060)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 193)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 13)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 208)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2061)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 208)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 13)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 209)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2061)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 209)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 14)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 224)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2062)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 224)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 14)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 225)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2062)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 225)]));
      Z_local[0] = (Z_local[0] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 15)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 240)]));
      Z_local[2] = (Z_local[2] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2063)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 240)]));
      Z_local[1] = (Z_local[1] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 15)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 241)]));
      Z_local[3] = (Z_local[3] + (X_shared[((((((int)threadIdx.x) >> 3) * 256) + (k_1 * 16)) + 2063)] * Y_shared[(((k_1 * 256) + ((((int)threadIdx.x) & 7) * 2)) + 241)]));
    }
  }
  Z[(((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2))] = Z_local[0];
  Z[((((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 256)] = Z_local[2];
  Z[((((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1)] = Z_local[1];
  Z[((((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 257)] = Z_local[3];
}

