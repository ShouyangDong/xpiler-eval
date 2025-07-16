
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[16];
  __shared__ float PadInput_shared[2592];
  __shared__ float placeholder_shared[288];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[8] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[9] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[10] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[11] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[12] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[13] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  depth_conv2d_nhwc_local[14] = 0.000000e+00f;
  depth_conv2d_nhwc_local[7] = 0.000000e+00f;
  depth_conv2d_nhwc_local[15] = 0.000000e+00f;
  float condval;
  if (((28 <= ((int)blockIdx.x)) && (2 <= (((int)blockIdx.x) % 28)))) {
    condval = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 7232)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  float condval_1;
  if ((28 <= ((int)blockIdx.x))) {
    condval_1 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 7168)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 32)] = condval_1;
  float condval_2;
  if ((28 <= ((int)blockIdx.x))) {
    condval_2 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 7104)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 64)] = condval_2;
  float condval_3;
  if ((28 <= ((int)blockIdx.x))) {
    condval_3 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 7040)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 96)] = condval_3;
  float condval_4;
  if ((28 <= ((int)blockIdx.x))) {
    condval_4 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 6976)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 128)] = condval_4;
  float condval_5;
  if ((28 <= ((int)blockIdx.x))) {
    condval_5 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 6912)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 160)] = condval_5;
  float condval_6;
  if ((28 <= ((int)blockIdx.x))) {
    condval_6 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 6848)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 192)] = condval_6;
  float condval_7;
  if ((28 <= ((int)blockIdx.x))) {
    condval_7 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 6784)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 224)] = condval_7;
  float condval_8;
  if ((28 <= ((int)blockIdx.x))) {
    condval_8 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 6720)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 256)] = condval_8;
  float condval_9;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_9 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) - 64)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 288)] = condval_9;
  PadInput_shared[(((int)threadIdx.x) + 320)] = placeholder[(((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x))];
  PadInput_shared[(((int)threadIdx.x) + 352)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 64)];
  PadInput_shared[(((int)threadIdx.x) + 384)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 128)];
  PadInput_shared[(((int)threadIdx.x) + 416)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 192)];
  PadInput_shared[(((int)threadIdx.x) + 448)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 256)];
  PadInput_shared[(((int)threadIdx.x) + 480)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 320)];
  PadInput_shared[(((int)threadIdx.x) + 512)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 384)];
  PadInput_shared[(((int)threadIdx.x) + 544)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 448)];
  float condval_10;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_10 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7104)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 576)] = condval_10;
  PadInput_shared[(((int)threadIdx.x) + 608)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7168)];
  PadInput_shared[(((int)threadIdx.x) + 640)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7232)];
  PadInput_shared[(((int)threadIdx.x) + 672)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7296)];
  PadInput_shared[(((int)threadIdx.x) + 704)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7360)];
  PadInput_shared[(((int)threadIdx.x) + 736)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7424)];
  PadInput_shared[(((int)threadIdx.x) + 768)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7488)];
  PadInput_shared[(((int)threadIdx.x) + 800)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7552)];
  PadInput_shared[(((int)threadIdx.x) + 832)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7616)];
  float condval_11;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_11 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14272)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 864)] = condval_11;
  PadInput_shared[(((int)threadIdx.x) + 896)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14336)];
  PadInput_shared[(((int)threadIdx.x) + 928)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14400)];
  PadInput_shared[(((int)threadIdx.x) + 960)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14464)];
  PadInput_shared[(((int)threadIdx.x) + 992)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14528)];
  PadInput_shared[(((int)threadIdx.x) + 1024)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14592)];
  PadInput_shared[(((int)threadIdx.x) + 1056)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14656)];
  PadInput_shared[(((int)threadIdx.x) + 1088)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14720)];
  PadInput_shared[(((int)threadIdx.x) + 1120)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14784)];
  float condval_12;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_12 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21440)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1152)] = condval_12;
  PadInput_shared[(((int)threadIdx.x) + 1184)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21504)];
  PadInput_shared[(((int)threadIdx.x) + 1216)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21568)];
  PadInput_shared[(((int)threadIdx.x) + 1248)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21632)];
  PadInput_shared[(((int)threadIdx.x) + 1280)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21696)];
  PadInput_shared[(((int)threadIdx.x) + 1312)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21760)];
  PadInput_shared[(((int)threadIdx.x) + 1344)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21824)];
  PadInput_shared[(((int)threadIdx.x) + 1376)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21888)];
  PadInput_shared[(((int)threadIdx.x) + 1408)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21952)];
  float condval_13;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_13 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 28608)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1440)] = condval_13;
  PadInput_shared[(((int)threadIdx.x) + 1472)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 28672)];
  PadInput_shared[(((int)threadIdx.x) + 1504)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 28736)];
  PadInput_shared[(((int)threadIdx.x) + 1536)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 28800)];
  PadInput_shared[(((int)threadIdx.x) + 1568)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 28864)];
  PadInput_shared[(((int)threadIdx.x) + 1600)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 28928)];
  PadInput_shared[(((int)threadIdx.x) + 1632)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 28992)];
  PadInput_shared[(((int)threadIdx.x) + 1664)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 29056)];
  PadInput_shared[(((int)threadIdx.x) + 1696)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 29120)];
  float condval_14;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_14 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 35776)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1728)] = condval_14;
  PadInput_shared[(((int)threadIdx.x) + 1760)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 35840)];
  PadInput_shared[(((int)threadIdx.x) + 1792)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 35904)];
  PadInput_shared[(((int)threadIdx.x) + 1824)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 35968)];
  PadInput_shared[(((int)threadIdx.x) + 1856)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 36032)];
  PadInput_shared[(((int)threadIdx.x) + 1888)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 36096)];
  PadInput_shared[(((int)threadIdx.x) + 1920)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 36160)];
  PadInput_shared[(((int)threadIdx.x) + 1952)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 36224)];
  PadInput_shared[(((int)threadIdx.x) + 1984)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 36288)];
  float condval_15;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_15 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 42944)];
  } else {
    condval_15 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2016)] = condval_15;
  PadInput_shared[(((int)threadIdx.x) + 2048)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43008)];
  PadInput_shared[(((int)threadIdx.x) + 2080)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43072)];
  PadInput_shared[(((int)threadIdx.x) + 2112)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43136)];
  PadInput_shared[(((int)threadIdx.x) + 2144)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43200)];
  PadInput_shared[(((int)threadIdx.x) + 2176)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43264)];
  PadInput_shared[(((int)threadIdx.x) + 2208)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43328)];
  PadInput_shared[(((int)threadIdx.x) + 2240)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43392)];
  PadInput_shared[(((int)threadIdx.x) + 2272)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 43456)];
  float condval_16;
  if ((2 <= (((int)blockIdx.x) % 28))) {
    condval_16 = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50112)];
  } else {
    condval_16 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2304)] = condval_16;
  PadInput_shared[(((int)threadIdx.x) + 2336)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50176)];
  PadInput_shared[(((int)threadIdx.x) + 2368)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50240)];
  PadInput_shared[(((int)threadIdx.x) + 2400)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50304)];
  PadInput_shared[(((int)threadIdx.x) + 2432)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50368)];
  PadInput_shared[(((int)threadIdx.x) + 2464)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50432)];
  PadInput_shared[(((int)threadIdx.x) + 2496)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50496)];
  PadInput_shared[(((int)threadIdx.x) + 2528)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50560)];
  PadInput_shared[(((int)threadIdx.x) + 2560)] = placeholder[((((((((int)blockIdx.x) / 28) * 57344) + (((((int)blockIdx.x) % 28) >> 1) * 512)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 50624)];
  int3 __1;
    int3 __2;
      int3 __3;
        int3 v_ = make_int3(0, 0, 0);
        int3 __4;
          int3 __5;
            int3 v__1 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
            int3 v__2 = make_int3(32, 32, 32);
            __5.x = (v__1.x/v__2.x);
            __5.y = (v__1.y/v__2.y);
            __5.z = (v__1.z/v__2.z);
          int3 v__3 = make_int3(64, 64, 64);
          __4.x = (__5.x*v__3.x);
          __4.y = (__5.y*v__3.y);
          __4.z = (__5.z*v__3.z);
        __3.x = (v_.x+__4.x);
        __3.y = (v_.y+__4.y);
        __3.z = (v_.z+__4.z);
      int3 v__4 = make_int3(((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32));
      __2.x = (__3.x+v__4.x);
      __2.y = (__3.y+v__4.y);
      __2.z = (__3.z+v__4.z);
    int3 __6;
      int3 v__5 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
      int3 v__6 = make_int3(32, 32, 32);
      __6.x = (v__5.x%v__6.x);
      __6.y = (v__5.y%v__6.y);
      __6.z = (v__5.z%v__6.z);
    __1.x = (__2.x+__6.x);
    __1.y = (__2.y+__6.y);
    __1.z = (__2.z+__6.z);
  *(float3*)(placeholder_shared + (((int)threadIdx.x) * 3)) = make_float3(placeholder_1[__1.x],placeholder_1[__1.y],placeholder_1[__1.z]);
  int3 __7;
    int3 __8;
      int3 __9;
        int3 v__7 = make_int3(192, 192, 192);
        int3 __10;
          int3 __11;
            int3 v__8 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
            int3 v__9 = make_int3(32, 32, 32);
            __11.x = (v__8.x/v__9.x);
            __11.y = (v__8.y/v__9.y);
            __11.z = (v__8.z/v__9.z);
          int3 v__10 = make_int3(64, 64, 64);
          __10.x = (__11.x*v__10.x);
          __10.y = (__11.y*v__10.y);
          __10.z = (__11.z*v__10.z);
        __9.x = (v__7.x+__10.x);
        __9.y = (v__7.y+__10.y);
        __9.z = (v__7.z+__10.z);
      int3 v__11 = make_int3(((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32));
      __8.x = (__9.x+v__11.x);
      __8.y = (__9.y+v__11.y);
      __8.z = (__9.z+v__11.z);
    int3 __12;
      int3 v__12 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
      int3 v__13 = make_int3(32, 32, 32);
      __12.x = (v__12.x%v__13.x);
      __12.y = (v__12.y%v__13.y);
      __12.z = (v__12.z%v__13.z);
    __7.x = (__8.x+__12.x);
    __7.y = (__8.y+__12.y);
    __7.z = (__8.z+__12.z);
  *(float3*)(placeholder_shared + ((((int)threadIdx.x) * 3) + 96)) = make_float3(placeholder_1[__7.x],placeholder_1[__7.y],placeholder_1[__7.z]);
  int3 __13;
    int3 __14;
      int3 __15;
        int3 v__14 = make_int3(384, 384, 384);
        int3 __16;
          int3 __17;
            int3 v__15 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
            int3 v__16 = make_int3(32, 32, 32);
            __17.x = (v__15.x/v__16.x);
            __17.y = (v__15.y/v__16.y);
            __17.z = (v__15.z/v__16.z);
          int3 v__17 = make_int3(64, 64, 64);
          __16.x = (__17.x*v__17.x);
          __16.y = (__17.y*v__17.y);
          __16.z = (__17.z*v__17.z);
        __15.x = (v__14.x+__16.x);
        __15.y = (v__14.y+__16.y);
        __15.z = (v__14.z+__16.z);
      int3 v__18 = make_int3(((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32), ((((int)blockIdx.x) & 1) * 32));
      __14.x = (__15.x+v__18.x);
      __14.y = (__15.y+v__18.y);
      __14.z = (__15.z+v__18.z);
    int3 __18;
      int3 v__19 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
      int3 v__20 = make_int3(32, 32, 32);
      __18.x = (v__19.x%v__20.x);
      __18.y = (v__19.y%v__20.y);
      __18.z = (v__19.z%v__20.z);
    __13.x = (__14.x+__18.x);
    __13.y = (__14.y+__18.y);
    __13.z = (__14.z+__18.z);
  *(float3*)(placeholder_shared + ((((int)threadIdx.x) * 3) + 192)) = make_float3(placeholder_1[__13.x],placeholder_1[__13.y],placeholder_1[__13.z]);
  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((int)threadIdx.x)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 128)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 32)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 160)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 64)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 192)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 288)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 416)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 320)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 448)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 352)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 480)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 576)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 704)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 608)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 736)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 640)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((int)threadIdx.x) + 768)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 64)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 192)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 96)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 224)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 128)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 256)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 352)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 480)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 384)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 512)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 416)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 544)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 640)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 768)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 672)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 800)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 704)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((int)threadIdx.x) + 832)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 576)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 704)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 608)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 736)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 640)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 768)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 864)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 992)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 896)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 1024)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 928)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 1056)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 1152)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 1280)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 1184)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 1312)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 1216)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((int)threadIdx.x) + 1344)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 640)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 768)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 672)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 800)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 704)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 832)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 928)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 1056)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 960)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 1088)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 992)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 1120)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 1216)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 1344)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 1248)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 1376)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 1280)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((int)threadIdx.x) + 1408)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1152)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1280)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1184)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1312)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1216)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1344)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1440)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1568)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1472)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1600)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1504)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1632)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1728)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1856)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1760)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1888)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1792)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((int)threadIdx.x) + 1920)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1216)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1344)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1248)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1376)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1280)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1408)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1504)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1632)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1536)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1664)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1568)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1696)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1792)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1920)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1824)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1952)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1856)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((int)threadIdx.x) + 1984)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 1728)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 1856)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 1760)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 1888)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 1792)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 1920)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2016)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 2144)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2048)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 2176)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2080)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 2208)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2304)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 2432)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2336)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 2464)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2368)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((int)threadIdx.x) + 2496)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 1792)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 1920)] * placeholder_shared[((int)threadIdx.x)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 1824)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 1952)] * placeholder_shared[(((int)threadIdx.x) + 32)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 1856)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 1984)] * placeholder_shared[(((int)threadIdx.x) + 64)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 2080)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 2208)] * placeholder_shared[(((int)threadIdx.x) + 96)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 2112)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 2240)] * placeholder_shared[(((int)threadIdx.x) + 128)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 2144)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 2272)] * placeholder_shared[(((int)threadIdx.x) + 160)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 2368)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 2496)] * placeholder_shared[(((int)threadIdx.x) + 192)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 2400)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 2528)] * placeholder_shared[(((int)threadIdx.x) + 224)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((int)threadIdx.x) + 2432)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((int)threadIdx.x) + 2560)] * placeholder_shared[(((int)threadIdx.x) + 256)]));
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 128)] = depth_conv2d_nhwc_local[8];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 64)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 192)] = depth_conv2d_nhwc_local[9];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3584)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3712)] = depth_conv2d_nhwc_local[10];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3648)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3776)] = depth_conv2d_nhwc_local[11];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7168)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7296)] = depth_conv2d_nhwc_local[12];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7232)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7360)] = depth_conv2d_nhwc_local[13];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10752)] = depth_conv2d_nhwc_local[6];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10880)] = depth_conv2d_nhwc_local[14];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10816)] = depth_conv2d_nhwc_local[7];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10944)] = depth_conv2d_nhwc_local[15];
}

