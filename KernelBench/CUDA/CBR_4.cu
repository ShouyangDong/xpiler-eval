
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
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float Conv2dOutput_local[32];
  __shared__ float PaddedInput_shared[222];
  __shared__ float kernel_shared[448];
  Conv2dOutput_local[0] = 0.000000e+00f;
  Conv2dOutput_local[16] = 0.000000e+00f;
  Conv2dOutput_local[1] = 0.000000e+00f;
  Conv2dOutput_local[17] = 0.000000e+00f;
  Conv2dOutput_local[2] = 0.000000e+00f;
  Conv2dOutput_local[18] = 0.000000e+00f;
  Conv2dOutput_local[3] = 0.000000e+00f;
  Conv2dOutput_local[19] = 0.000000e+00f;
  Conv2dOutput_local[4] = 0.000000e+00f;
  Conv2dOutput_local[20] = 0.000000e+00f;
  Conv2dOutput_local[5] = 0.000000e+00f;
  Conv2dOutput_local[21] = 0.000000e+00f;
  Conv2dOutput_local[6] = 0.000000e+00f;
  Conv2dOutput_local[22] = 0.000000e+00f;
  Conv2dOutput_local[7] = 0.000000e+00f;
  Conv2dOutput_local[23] = 0.000000e+00f;
  Conv2dOutput_local[8] = 0.000000e+00f;
  Conv2dOutput_local[24] = 0.000000e+00f;
  Conv2dOutput_local[9] = 0.000000e+00f;
  Conv2dOutput_local[25] = 0.000000e+00f;
  Conv2dOutput_local[10] = 0.000000e+00f;
  Conv2dOutput_local[26] = 0.000000e+00f;
  Conv2dOutput_local[11] = 0.000000e+00f;
  Conv2dOutput_local[27] = 0.000000e+00f;
  Conv2dOutput_local[12] = 0.000000e+00f;
  Conv2dOutput_local[28] = 0.000000e+00f;
  Conv2dOutput_local[13] = 0.000000e+00f;
  Conv2dOutput_local[29] = 0.000000e+00f;
  Conv2dOutput_local[14] = 0.000000e+00f;
  Conv2dOutput_local[30] = 0.000000e+00f;
  Conv2dOutput_local[15] = 0.000000e+00f;
  Conv2dOutput_local[31] = 0.000000e+00f;
  for (int ry_0 = 0; ry_0 < 7; ++ry_0) {
    for (int rc_0 = 0; rc_0 < 3; ++rc_0) {
      __syncthreads();
      if (((int)threadIdx.x) < 74) {
        float condval;
        if (((((3 <= (((((((int)blockIdx.x) % 392) / 7) * 4) + (((((int)threadIdx.x) % 37) * 3) / 37)) + ry_0)) && ((((((((int)blockIdx.x) % 392) / 7) * 4) + (((((int)threadIdx.x) % 37) * 3) / 37)) + ry_0) < 227)) && (3 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) * 3) % 37)))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) * 3) % 37)) < 227))) {
          condval = data[((((((((((((int)blockIdx.x) / 392) * 301056) + ((((int)threadIdx.x) / 37) * 150528)) + (((((int)blockIdx.x) % 392) / 7) * 2688)) + ((((((int)threadIdx.x) % 37) * 3) / 37) * 672)) + (ry_0 * 672)) + ((((int)blockIdx.x) % 7) * 96)) + (((((int)threadIdx.x) * 3) % 37) * 3)) + rc_0) - 2025)];
        } else {
          condval = 0.000000e+00f;
        }
        PaddedInput_shared[((((((int)threadIdx.x) / 37) * 111) + ((((((int)threadIdx.x) % 37) * 3) / 37) * 37)) + ((((int)threadIdx.x) * 3) % 37))] = condval;
        float condval_1;
        if (((((3 <= (((((((int)blockIdx.x) % 392) / 7) * 4) + ((((((int)threadIdx.x) % 37) * 3) + 1) / 37)) + ry_0)) && ((((((((int)blockIdx.x) % 392) / 7) * 4) + ((((((int)threadIdx.x) % 37) * 3) + 1) / 37)) + ry_0) < 227)) && (3 <= (((((int)blockIdx.x) % 7) * 32) + (((((int)threadIdx.x) * 3) + 1) % 37)))) && ((((((int)blockIdx.x) % 7) * 32) + (((((int)threadIdx.x) * 3) + 1) % 37)) < 227))) {
          condval_1 = data[((((((((((((int)blockIdx.x) / 392) * 301056) + ((((int)threadIdx.x) / 37) * 150528)) + (((((int)blockIdx.x) % 392) / 7) * 2688)) + (((((((int)threadIdx.x) % 37) * 3) + 1) / 37) * 672)) + (ry_0 * 672)) + ((((int)blockIdx.x) % 7) * 96)) + ((((((int)threadIdx.x) * 3) + 1) % 37) * 3)) + rc_0) - 2025)];
        } else {
          condval_1 = 0.000000e+00f;
        }
        PaddedInput_shared[((((((int)threadIdx.x) / 37) * 111) + (((((((int)threadIdx.x) % 37) * 3) + 1) / 37) * 37)) + (((((int)threadIdx.x) * 3) + 1) % 37))] = condval_1;
        float condval_2;
        if (((((3 <= (((((((int)blockIdx.x) % 392) / 7) * 4) + ((((((int)threadIdx.x) % 37) * 3) + 2) / 37)) + ry_0)) && ((((((((int)blockIdx.x) % 392) / 7) * 4) + ((((((int)threadIdx.x) % 37) * 3) + 2) / 37)) + ry_0) < 227)) && (3 <= (((((int)blockIdx.x) % 7) * 32) + (((((int)threadIdx.x) * 3) + 2) % 37)))) && ((((((int)blockIdx.x) % 7) * 32) + (((((int)threadIdx.x) * 3) + 2) % 37)) < 227))) {
          condval_2 = data[((((((((((((int)blockIdx.x) / 392) * 301056) + ((((int)threadIdx.x) / 37) * 150528)) + (((((int)blockIdx.x) % 392) / 7) * 2688)) + (((((((int)threadIdx.x) % 37) * 3) + 2) / 37) * 672)) + (ry_0 * 672)) + ((((int)blockIdx.x) % 7) * 96)) + ((((((int)threadIdx.x) * 3) + 2) % 37) * 3)) + rc_0) - 2025)];
        } else {
          condval_2 = 0.000000e+00f;
        }
        PaddedInput_shared[((((((int)threadIdx.x) / 37) * 111) + (((((((int)threadIdx.x) % 37) * 3) + 2) / 37) * 37)) + (((((int)threadIdx.x) * 3) + 2) % 37))] = condval_2;
      }
      *(float2*)(kernel_shared + (((int)threadIdx.x) * 2)) = *(float2*)(kernel + ((((ry_0 * 1344) + ((((int)threadIdx.x) >> 5) * 192)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)));
      if (((int)threadIdx.x) < 96) {
        *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(kernel + (((((ry_0 * 1344) + ((((int)threadIdx.x) >> 5) * 192)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 768));
      }
      __syncthreads();
      Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[((((int)threadIdx.x) >> 6) * 16)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[16] = (Conv2dOutput_local[16] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 111)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 2)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[17] = (Conv2dOutput_local[17] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 113)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 4)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[18] = (Conv2dOutput_local[18] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 115)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 6)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[19] = (Conv2dOutput_local[19] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 117)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 8)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[20] = (Conv2dOutput_local[20] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 119)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 10)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[21] = (Conv2dOutput_local[21] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 121)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 12)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[22] = (Conv2dOutput_local[22] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 123)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 14)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[23] = (Conv2dOutput_local[23] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 125)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[8] = (Conv2dOutput_local[8] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 74)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[24] = (Conv2dOutput_local[24] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 185)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[9] = (Conv2dOutput_local[9] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 76)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[25] = (Conv2dOutput_local[25] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 187)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[10] = (Conv2dOutput_local[10] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 78)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[26] = (Conv2dOutput_local[26] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 189)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[11] = (Conv2dOutput_local[11] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 80)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[27] = (Conv2dOutput_local[27] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 191)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[12] = (Conv2dOutput_local[12] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 82)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[28] = (Conv2dOutput_local[28] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 193)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[13] = (Conv2dOutput_local[13] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 84)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[29] = (Conv2dOutput_local[29] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 195)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[14] = (Conv2dOutput_local[14] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 86)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[30] = (Conv2dOutput_local[30] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 197)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[15] = (Conv2dOutput_local[15] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 88)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[31] = (Conv2dOutput_local[31] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 199)] * kernel_shared[(((int)threadIdx.x) & 63)]));
      Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 1)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[16] = (Conv2dOutput_local[16] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 112)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 3)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[17] = (Conv2dOutput_local[17] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 114)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 5)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[18] = (Conv2dOutput_local[18] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 116)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 7)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[19] = (Conv2dOutput_local[19] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 118)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 9)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[20] = (Conv2dOutput_local[20] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 120)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 11)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[21] = (Conv2dOutput_local[21] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 122)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 13)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[22] = (Conv2dOutput_local[22] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 124)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 15)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[23] = (Conv2dOutput_local[23] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 126)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[8] = (Conv2dOutput_local[8] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 75)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[24] = (Conv2dOutput_local[24] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 186)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[9] = (Conv2dOutput_local[9] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 77)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[25] = (Conv2dOutput_local[25] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 188)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[10] = (Conv2dOutput_local[10] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 79)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[26] = (Conv2dOutput_local[26] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 190)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[11] = (Conv2dOutput_local[11] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 81)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[27] = (Conv2dOutput_local[27] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 192)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[12] = (Conv2dOutput_local[12] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 83)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[28] = (Conv2dOutput_local[28] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 194)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[13] = (Conv2dOutput_local[13] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 85)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[29] = (Conv2dOutput_local[29] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 196)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[14] = (Conv2dOutput_local[14] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 87)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[30] = (Conv2dOutput_local[30] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 198)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[15] = (Conv2dOutput_local[15] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 89)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[31] = (Conv2dOutput_local[31] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 200)] * kernel_shared[((((int)threadIdx.x) & 63) + 64)]));
      Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 2)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[16] = (Conv2dOutput_local[16] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 113)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 4)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[17] = (Conv2dOutput_local[17] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 115)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 6)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[18] = (Conv2dOutput_local[18] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 117)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 8)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[19] = (Conv2dOutput_local[19] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 119)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 10)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[20] = (Conv2dOutput_local[20] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 121)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 12)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[21] = (Conv2dOutput_local[21] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 123)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 14)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[22] = (Conv2dOutput_local[22] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 125)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 16)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[23] = (Conv2dOutput_local[23] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 127)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[8] = (Conv2dOutput_local[8] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 76)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[24] = (Conv2dOutput_local[24] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 187)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[9] = (Conv2dOutput_local[9] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 78)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[25] = (Conv2dOutput_local[25] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 189)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[10] = (Conv2dOutput_local[10] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 80)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[26] = (Conv2dOutput_local[26] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 191)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[11] = (Conv2dOutput_local[11] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 82)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[27] = (Conv2dOutput_local[27] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 193)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[12] = (Conv2dOutput_local[12] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 84)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[28] = (Conv2dOutput_local[28] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 195)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[13] = (Conv2dOutput_local[13] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 86)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[29] = (Conv2dOutput_local[29] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 197)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[14] = (Conv2dOutput_local[14] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 88)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[30] = (Conv2dOutput_local[30] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 199)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[15] = (Conv2dOutput_local[15] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 90)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[31] = (Conv2dOutput_local[31] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 201)] * kernel_shared[((((int)threadIdx.x) & 63) + 128)]));
      Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 3)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[16] = (Conv2dOutput_local[16] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 114)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 5)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[17] = (Conv2dOutput_local[17] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 116)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 7)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[18] = (Conv2dOutput_local[18] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 118)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 9)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[19] = (Conv2dOutput_local[19] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 120)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 11)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[20] = (Conv2dOutput_local[20] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 122)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 13)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[21] = (Conv2dOutput_local[21] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 124)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 15)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[22] = (Conv2dOutput_local[22] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 126)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 17)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[23] = (Conv2dOutput_local[23] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 128)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[8] = (Conv2dOutput_local[8] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 77)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[24] = (Conv2dOutput_local[24] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 188)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[9] = (Conv2dOutput_local[9] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 79)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[25] = (Conv2dOutput_local[25] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 190)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[10] = (Conv2dOutput_local[10] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 81)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[26] = (Conv2dOutput_local[26] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 192)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[11] = (Conv2dOutput_local[11] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 83)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[27] = (Conv2dOutput_local[27] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 194)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[12] = (Conv2dOutput_local[12] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 85)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[28] = (Conv2dOutput_local[28] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 196)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[13] = (Conv2dOutput_local[13] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 87)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[29] = (Conv2dOutput_local[29] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 198)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[14] = (Conv2dOutput_local[14] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 89)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[30] = (Conv2dOutput_local[30] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 200)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[15] = (Conv2dOutput_local[15] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 91)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[31] = (Conv2dOutput_local[31] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 202)] * kernel_shared[((((int)threadIdx.x) & 63) + 192)]));
      Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 4)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[16] = (Conv2dOutput_local[16] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 115)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 6)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[17] = (Conv2dOutput_local[17] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 117)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 8)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[18] = (Conv2dOutput_local[18] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 119)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 10)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[19] = (Conv2dOutput_local[19] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 121)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 12)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[20] = (Conv2dOutput_local[20] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 123)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 14)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[21] = (Conv2dOutput_local[21] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 125)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 16)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[22] = (Conv2dOutput_local[22] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 127)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 18)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[23] = (Conv2dOutput_local[23] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 129)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[8] = (Conv2dOutput_local[8] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 78)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[24] = (Conv2dOutput_local[24] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 189)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[9] = (Conv2dOutput_local[9] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 80)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[25] = (Conv2dOutput_local[25] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 191)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[10] = (Conv2dOutput_local[10] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 82)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[26] = (Conv2dOutput_local[26] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 193)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[11] = (Conv2dOutput_local[11] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 84)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[27] = (Conv2dOutput_local[27] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 195)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[12] = (Conv2dOutput_local[12] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 86)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[28] = (Conv2dOutput_local[28] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 197)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[13] = (Conv2dOutput_local[13] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 88)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[29] = (Conv2dOutput_local[29] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 199)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[14] = (Conv2dOutput_local[14] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 90)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[30] = (Conv2dOutput_local[30] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 201)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[15] = (Conv2dOutput_local[15] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 92)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[31] = (Conv2dOutput_local[31] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 203)] * kernel_shared[((((int)threadIdx.x) & 63) + 256)]));
      Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 5)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[16] = (Conv2dOutput_local[16] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 116)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 7)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[17] = (Conv2dOutput_local[17] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 118)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 9)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[18] = (Conv2dOutput_local[18] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 120)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 11)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[19] = (Conv2dOutput_local[19] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 122)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 13)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[20] = (Conv2dOutput_local[20] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 124)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 15)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[21] = (Conv2dOutput_local[21] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 126)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 17)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[22] = (Conv2dOutput_local[22] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 128)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 19)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[23] = (Conv2dOutput_local[23] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 130)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[8] = (Conv2dOutput_local[8] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 79)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[24] = (Conv2dOutput_local[24] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 190)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[9] = (Conv2dOutput_local[9] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 81)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[25] = (Conv2dOutput_local[25] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 192)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[10] = (Conv2dOutput_local[10] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 83)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[26] = (Conv2dOutput_local[26] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 194)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[11] = (Conv2dOutput_local[11] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 85)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[27] = (Conv2dOutput_local[27] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 196)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[12] = (Conv2dOutput_local[12] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 87)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[28] = (Conv2dOutput_local[28] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 198)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[13] = (Conv2dOutput_local[13] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 89)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[29] = (Conv2dOutput_local[29] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 200)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[14] = (Conv2dOutput_local[14] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 91)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[30] = (Conv2dOutput_local[30] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 202)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[15] = (Conv2dOutput_local[15] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 93)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[31] = (Conv2dOutput_local[31] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 204)] * kernel_shared[((((int)threadIdx.x) & 63) + 320)]));
      Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 6)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[16] = (Conv2dOutput_local[16] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 117)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 8)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[17] = (Conv2dOutput_local[17] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 119)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 10)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[18] = (Conv2dOutput_local[18] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 121)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 12)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[19] = (Conv2dOutput_local[19] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 123)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 14)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[20] = (Conv2dOutput_local[20] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 125)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 16)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[21] = (Conv2dOutput_local[21] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 127)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 18)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[22] = (Conv2dOutput_local[22] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 129)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 20)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[23] = (Conv2dOutput_local[23] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 131)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[8] = (Conv2dOutput_local[8] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 80)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[24] = (Conv2dOutput_local[24] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 191)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[9] = (Conv2dOutput_local[9] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 82)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[25] = (Conv2dOutput_local[25] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 193)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[10] = (Conv2dOutput_local[10] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 84)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[26] = (Conv2dOutput_local[26] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 195)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[11] = (Conv2dOutput_local[11] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 86)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[27] = (Conv2dOutput_local[27] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 197)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[12] = (Conv2dOutput_local[12] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 88)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[28] = (Conv2dOutput_local[28] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 199)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[13] = (Conv2dOutput_local[13] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 90)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[29] = (Conv2dOutput_local[29] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 201)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[14] = (Conv2dOutput_local[14] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 92)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[30] = (Conv2dOutput_local[30] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 203)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[15] = (Conv2dOutput_local[15] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 94)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
      Conv2dOutput_local[31] = (Conv2dOutput_local[31] + (PaddedInput_shared[(((((int)threadIdx.x) >> 6) * 16) + 205)] * kernel_shared[((((int)threadIdx.x) & 63) + 384)]));
    }
  }
  compute[((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63))] = max((((Conv2dOutput_local[0] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 802816)] = max((((Conv2dOutput_local[16] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 64)] = max((((Conv2dOutput_local[1] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 802880)] = max((((Conv2dOutput_local[17] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 128)] = max((((Conv2dOutput_local[2] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 802944)] = max((((Conv2dOutput_local[18] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 192)] = max((((Conv2dOutput_local[3] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 803008)] = max((((Conv2dOutput_local[19] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 256)] = max((((Conv2dOutput_local[4] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 803072)] = max((((Conv2dOutput_local[20] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 320)] = max((((Conv2dOutput_local[5] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 803136)] = max((((Conv2dOutput_local[21] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 384)] = max((((Conv2dOutput_local[6] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 803200)] = max((((Conv2dOutput_local[22] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 448)] = max((((Conv2dOutput_local[7] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 803264)] = max((((Conv2dOutput_local[23] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7168)] = max((((Conv2dOutput_local[8] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 809984)] = max((((Conv2dOutput_local[24] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7232)] = max((((Conv2dOutput_local[9] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 810048)] = max((((Conv2dOutput_local[25] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7296)] = max((((Conv2dOutput_local[10] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 810112)] = max((((Conv2dOutput_local[26] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7360)] = max((((Conv2dOutput_local[11] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 810176)] = max((((Conv2dOutput_local[27] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7424)] = max((((Conv2dOutput_local[12] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 810240)] = max((((Conv2dOutput_local[28] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7488)] = max((((Conv2dOutput_local[13] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 810304)] = max((((Conv2dOutput_local[29] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7552)] = max((((Conv2dOutput_local[14] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 810368)] = max((((Conv2dOutput_local[30] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 7616)] = max((((Conv2dOutput_local[15] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 392) * 1605632) + (((((int)blockIdx.x) % 392) / 7) * 14336)) + ((((int)blockIdx.x) % 7) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 810432)] = max((((Conv2dOutput_local[31] + bias[(((int)threadIdx.x) & 63)]) * bn_scale[(((int)threadIdx.x) & 63)]) + bn_offset[(((int)threadIdx.x) & 63)]), 0.000000e+00f);
}

