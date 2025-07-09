
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
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[16];
  __shared__ float PadInput_shared[9248];
  __shared__ float placeholder_shared[288];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[8] = 0.000000e+00f;
  depth_conv2d_nhwc_local[9] = 0.000000e+00f;
  depth_conv2d_nhwc_local[10] = 0.000000e+00f;
  depth_conv2d_nhwc_local[11] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  depth_conv2d_nhwc_local[7] = 0.000000e+00f;
  depth_conv2d_nhwc_local[12] = 0.000000e+00f;
  depth_conv2d_nhwc_local[13] = 0.000000e+00f;
  depth_conv2d_nhwc_local[14] = 0.000000e+00f;
  depth_conv2d_nhwc_local[15] = 0.000000e+00f;
  float condval;
  if (((14 <= ((int)blockIdx.x)) && (1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((int)threadIdx.x) >> 5))))) {
    condval = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  float condval_1;
  if ((14 <= ((int)blockIdx.x))) {
    condval_1 = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6976)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 128)] = condval_1;
  float condval_2;
  if ((14 <= ((int)blockIdx.x))) {
    condval_2 = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6720)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 256)] = condval_2;
  float condval_3;
  if ((14 <= ((int)blockIdx.x))) {
    condval_3 = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6464)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 384)] = condval_3;
  float condval_4;
  if (((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((int)threadIdx.x) + 512) / 544))) && (1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 17))))) {
    condval_4 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 512) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 16) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 512)] = condval_4;
  PadInput_shared[(((int)threadIdx.x) + 640)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 640) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7040)];
  PadInput_shared[(((int)threadIdx.x) + 768)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 768) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6784)];
  PadInput_shared[(((int)threadIdx.x) + 896)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 896) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6528)];
  float condval_5;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 15) % 17)))) {
    condval_5 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1024) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 15) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1024)] = condval_5;
  PadInput_shared[(((int)threadIdx.x) + 1152)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1152) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7104)];
  PadInput_shared[(((int)threadIdx.x) + 1280)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1280) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6848)];
  PadInput_shared[(((int)threadIdx.x) + 1408)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1408) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6592)];
  float condval_6;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 14) % 17)))) {
    condval_6 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1536) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 14) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1536)] = condval_6;
  PadInput_shared[(((int)threadIdx.x) + 1664)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1664) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7168)];
  PadInput_shared[(((int)threadIdx.x) + 1792)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1792) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6912)];
  PadInput_shared[(((int)threadIdx.x) + 1920)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 1920) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6656)];
  PadInput_shared[(((int)threadIdx.x) + 2048)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 2048) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6400)];
  float condval_7;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((int)threadIdx.x) >> 5)))) {
    condval_7 = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 21440)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2176)] = condval_7;
  PadInput_shared[(((int)threadIdx.x) + 2304)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 2304) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6976)];
  PadInput_shared[(((int)threadIdx.x) + 2432)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 2432) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6720)];
  PadInput_shared[(((int)threadIdx.x) + 2560)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 2560) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6464)];
  float condval_8;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 17)))) {
    condval_8 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 2688) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 16) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2688)] = condval_8;
  PadInput_shared[(((int)threadIdx.x) + 2816)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 2816) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7040)];
  PadInput_shared[(((int)threadIdx.x) + 2944)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 2944) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6784)];
  PadInput_shared[(((int)threadIdx.x) + 3072)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3072) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6528)];
  float condval_9;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 15) % 17)))) {
    condval_9 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3200) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 15) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3200)] = condval_9;
  PadInput_shared[(((int)threadIdx.x) + 3328)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3328) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7104)];
  PadInput_shared[(((int)threadIdx.x) + 3456)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3456) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6848)];
  PadInput_shared[(((int)threadIdx.x) + 3584)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3584) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6592)];
  float condval_10;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 14) % 17)))) {
    condval_10 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3712) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 14) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3712)] = condval_10;
  PadInput_shared[(((int)threadIdx.x) + 3840)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3840) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7168)];
  PadInput_shared[(((int)threadIdx.x) + 3968)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 3968) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6912)];
  PadInput_shared[(((int)threadIdx.x) + 4096)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 4096) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6656)];
  PadInput_shared[(((int)threadIdx.x) + 4224)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 4224) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6400)];
  float condval_11;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((int)threadIdx.x) >> 5)))) {
    condval_11 = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 50112)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 4352)] = condval_11;
  PadInput_shared[(((int)threadIdx.x) + 4480)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 4480) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6976)];
  PadInput_shared[(((int)threadIdx.x) + 4608)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 4608) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6720)];
  PadInput_shared[(((int)threadIdx.x) + 4736)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 4736) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6464)];
  float condval_12;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 17)))) {
    condval_12 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 4864) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 16) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 4864)] = condval_12;
  PadInput_shared[(((int)threadIdx.x) + 4992)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 4992) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7040)];
  PadInput_shared[(((int)threadIdx.x) + 5120)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 5120) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6784)];
  PadInput_shared[(((int)threadIdx.x) + 5248)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 5248) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6528)];
  float condval_13;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 15) % 17)))) {
    condval_13 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 5376) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 15) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 5376)] = condval_13;
  PadInput_shared[(((int)threadIdx.x) + 5504)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 5504) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7104)];
  PadInput_shared[(((int)threadIdx.x) + 5632)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 5632) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6848)];
  PadInput_shared[(((int)threadIdx.x) + 5760)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 5760) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6592)];
  float condval_14;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 14) % 17)))) {
    condval_14 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 5888) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 14) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 5888)] = condval_14;
  PadInput_shared[(((int)threadIdx.x) + 6016)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 6016) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7168)];
  PadInput_shared[(((int)threadIdx.x) + 6144)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 6144) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6912)];
  PadInput_shared[(((int)threadIdx.x) + 6272)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 6272) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6656)];
  PadInput_shared[(((int)threadIdx.x) + 6400)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 6400) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6400)];
  float condval_15;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((int)threadIdx.x) >> 5)))) {
    condval_15 = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 78784)];
  } else {
    condval_15 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 6528)] = condval_15;
  PadInput_shared[(((int)threadIdx.x) + 6656)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 6656) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6976)];
  PadInput_shared[(((int)threadIdx.x) + 6784)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 6784) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6720)];
  PadInput_shared[(((int)threadIdx.x) + 6912)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 6912) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6464)];
  float condval_16;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 17)))) {
    condval_16 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7040) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 16) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_16 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 7040)] = condval_16;
  PadInput_shared[(((int)threadIdx.x) + 7168)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7168) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7040)];
  PadInput_shared[(((int)threadIdx.x) + 7296)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7296) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6784)];
  PadInput_shared[(((int)threadIdx.x) + 7424)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7424) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6528)];
  float condval_17;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 15) % 17)))) {
    condval_17 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7552) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 15) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_17 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 7552)] = condval_17;
  PadInput_shared[(((int)threadIdx.x) + 7680)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7680) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7104)];
  PadInput_shared[(((int)threadIdx.x) + 7808)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7808) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6848)];
  PadInput_shared[(((int)threadIdx.x) + 7936)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 7936) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6592)];
  float condval_18;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((((int)threadIdx.x) >> 5) + 14) % 17)))) {
    condval_18 = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 8064) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((((int)threadIdx.x) >> 5) + 14) % 17) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7232)];
  } else {
    condval_18 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 8064)] = condval_18;
  PadInput_shared[(((int)threadIdx.x) + 8192)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 8192) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 7168)];
  PadInput_shared[(((int)threadIdx.x) + 8320)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 8320) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6912)];
  PadInput_shared[(((int)threadIdx.x) + 8448)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 8448) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6656)];
  PadInput_shared[(((int)threadIdx.x) + 8576)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 8576) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6400)];
  float condval_19;
  if ((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 16) + (((int)threadIdx.x) >> 5)))) {
    condval_19 = placeholder[(((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 107456)];
  } else {
    condval_19 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 8704)] = condval_19;
  PadInput_shared[(((int)threadIdx.x) + 8832)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 8832) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6976)];
  PadInput_shared[(((int)threadIdx.x) + 8960)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 8960) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6720)];
  PadInput_shared[(((int)threadIdx.x) + 9088)] = placeholder[((((((((((int)blockIdx.x) / 14) * 114688) + (((((int)threadIdx.x) + 9088) / 544) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) - 6464)];
  if (((int)threadIdx.x) < 32) {
    PadInput_shared[(((int)threadIdx.x) + 9216)] = placeholder[((((((((int)blockIdx.x) / 14) * 114688) + (((((int)blockIdx.x) % 14) >> 1) * 1024)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 108480)];
  }
  if (((int)threadIdx.x) < 96) {
      int3 __1;
        int3 __2;
          int3 v_ = make_int3(((((int)threadIdx.x) >> 5) * 96), ((((int)threadIdx.x) >> 5) * 96), ((((int)threadIdx.x) >> 5) * 96));
          int3 __3;
            int3 __4;
              int3 v__1 = make_int3((((((int)threadIdx.x) & 31) * 3))+(1*0), (((((int)threadIdx.x) & 31) * 3))+(1*1), (((((int)threadIdx.x) & 31) * 3))+(1*2));
              int3 v__2 = make_int3(32, 32, 32);
              __4.x = (v__1.x/v__2.x);
              __4.y = (v__1.y/v__2.y);
              __4.z = (v__1.z/v__2.z);
            int3 v__3 = make_int3(32, 32, 32);
            __3.x = (__4.x*v__3.x);
            __3.y = (__4.y*v__3.y);
            __3.z = (__4.z*v__3.z);
          __2.x = (v_.x+__3.x);
          __2.y = (v_.y+__3.y);
          __2.z = (v_.z+__3.z);
        int3 __5;
          int3 v__4 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
          int3 v__5 = make_int3(32, 32, 32);
          __5.x = (v__4.x%v__5.x);
          __5.y = (v__4.y%v__5.y);
          __5.z = (v__4.z%v__5.z);
        __1.x = (__2.x+__5.x);
        __1.y = (__2.y+__5.y);
        __1.z = (__2.z+__5.z);
      int3 __6;
        int3 __7;
          int3 __8;
            int3 v__6 = make_int3(((((int)threadIdx.x) >> 5) * 192), ((((int)threadIdx.x) >> 5) * 192), ((((int)threadIdx.x) >> 5) * 192));
            int3 __9;
              int3 __10;
                int3 v__7 = make_int3((((((int)threadIdx.x) & 31) * 3))+(1*0), (((((int)threadIdx.x) & 31) * 3))+(1*1), (((((int)threadIdx.x) & 31) * 3))+(1*2));
                int3 v__8 = make_int3(32, 32, 32);
                __10.x = (v__7.x/v__8.x);
                __10.y = (v__7.y/v__8.y);
                __10.z = (v__7.z/v__8.z);
              int3 v__9 = make_int3(64, 64, 64);
              __9.x = (__10.x*v__9.x);
              __9.y = (__10.y*v__9.y);
              __9.z = (__10.z*v__9.z);
            __8.x = (v__6.x+__9.x);
            __8.y = (v__6.y+__9.y);
            __8.z = (v__6.z+__9.z);
          int3 v__10 = make_int3(((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32));
          __7.x = (__8.x+v__10.x);
          __7.y = (__8.y+v__10.y);
          __7.z = (__8.z+v__10.z);
        int3 __11;
          int3 v__11 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
          int3 v__12 = make_int3(32, 32, 32);
          __11.x = (v__11.x%v__12.x);
          __11.y = (v__11.y%v__12.y);
          __11.z = (v__11.z%v__12.z);
        __6.x = (__7.x+__11.x);
        __6.y = (__7.y+__11.y);
        __6.z = (__7.z+__11.z);
      float3 v__13 = make_float3(placeholder_1[__6.x],placeholder_1[__6.y],placeholder_1[__6.z]);
      placeholder_shared[__1.x] = v__13.x;
      placeholder_shared[__1.y] = v__13.y;
      placeholder_shared[__1.z] = v__13.z;
  }
  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31))] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 64)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 128)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 192)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1152)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1216)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 32)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 96)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 160)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 224)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1184)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1248)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 64)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 192)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 256)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1152)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1216)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 544)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 608)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1632)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1760)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1824)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 576)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 640)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 768)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1728)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1792)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1856)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 608)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 800)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1760)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1824)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1888)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1152)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1216)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2176)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1184)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1248)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2208)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2400)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1152)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1216)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2432)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 256)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 320)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 448)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1472)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1536)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 288)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 352)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 480)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1440)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1504)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1568)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 320)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 448)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 512)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1472)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1536)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1600)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 800)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 864)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 928)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1888)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2080)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 832)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 896)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 960)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1920)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2112)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 864)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 928)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2080)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2144)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1472)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1536)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2432)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2496)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2560)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1440)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1504)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1568)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2464)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2528)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2592)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2656)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1472)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1536)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 1600)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2496)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2560)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 2176) + (((int)threadIdx.x) & 31)) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 64)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 128)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 192)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 256)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 320)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 384)] = depth_conv2d_nhwc_local[6];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 448)] = depth_conv2d_nhwc_local[7];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3584)] = depth_conv2d_nhwc_local[8];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3648)] = depth_conv2d_nhwc_local[9];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3712)] = depth_conv2d_nhwc_local[10];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3776)] = depth_conv2d_nhwc_local[11];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3840)] = depth_conv2d_nhwc_local[12];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3904)] = depth_conv2d_nhwc_local[13];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3968)] = depth_conv2d_nhwc_local[14];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 14) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 4032)] = depth_conv2d_nhwc_local[15];
}

