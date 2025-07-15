
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
  float depth_conv2d_nhwc_local[32];
  __shared__ float PadInput_shared[5760];
  __shared__ float placeholder_shared[288];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[8] = 0.000000e+00f;
  depth_conv2d_nhwc_local[12] = 0.000000e+00f;
  depth_conv2d_nhwc_local[16] = 0.000000e+00f;
  depth_conv2d_nhwc_local[20] = 0.000000e+00f;
  depth_conv2d_nhwc_local[24] = 0.000000e+00f;
  depth_conv2d_nhwc_local[28] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[9] = 0.000000e+00f;
  depth_conv2d_nhwc_local[13] = 0.000000e+00f;
  depth_conv2d_nhwc_local[17] = 0.000000e+00f;
  depth_conv2d_nhwc_local[21] = 0.000000e+00f;
  depth_conv2d_nhwc_local[25] = 0.000000e+00f;
  depth_conv2d_nhwc_local[29] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  depth_conv2d_nhwc_local[10] = 0.000000e+00f;
  depth_conv2d_nhwc_local[14] = 0.000000e+00f;
  depth_conv2d_nhwc_local[18] = 0.000000e+00f;
  depth_conv2d_nhwc_local[22] = 0.000000e+00f;
  depth_conv2d_nhwc_local[26] = 0.000000e+00f;
  depth_conv2d_nhwc_local[30] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[7] = 0.000000e+00f;
  depth_conv2d_nhwc_local[11] = 0.000000e+00f;
  depth_conv2d_nhwc_local[15] = 0.000000e+00f;
  depth_conv2d_nhwc_local[19] = 0.000000e+00f;
  depth_conv2d_nhwc_local[23] = 0.000000e+00f;
  depth_conv2d_nhwc_local[27] = 0.000000e+00f;
  depth_conv2d_nhwc_local[31] = 0.000000e+00f;
  float condval;
  if (((14 <= ((int)blockIdx.x)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5))))) {
    condval = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3616)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  float condval_1;
  if ((14 <= ((int)blockIdx.x))) {
    condval_1 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 128)] = condval_1;
  float condval_2;
  if ((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((int)threadIdx.x) + 256) / 320))) && (1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_2 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 256) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 256)] = condval_2;
  PadInput_shared[(((int)threadIdx.x) + 384)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 384) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_3;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_3 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 512) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 512)] = condval_3;
  float condval_4;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_4 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 3552)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 640)] = condval_4;
  PadInput_shared[(((int)threadIdx.x) + 768)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 768) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_5;
  if (((1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_5 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 896) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 896)] = condval_5;
  PadInput_shared[(((int)threadIdx.x) + 1024)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 1024) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_6;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_6 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 1152) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1152)] = condval_6;
  float condval_7;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_7 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 10720)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1280)] = condval_7;
  PadInput_shared[(((int)threadIdx.x) + 1408)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 1408) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_8;
  if (((1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_8 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 1536) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1536)] = condval_8;
  PadInput_shared[(((int)threadIdx.x) + 1664)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 1664) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_9;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_9 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 1792) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1792)] = condval_9;
  float condval_10;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_10 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 17888)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1920)] = condval_10;
  PadInput_shared[(((int)threadIdx.x) + 2048)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 2048) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_11;
  if (((1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_11 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 2176) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2176)] = condval_11;
  PadInput_shared[(((int)threadIdx.x) + 2304)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 2304) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_12;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_12 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 2432) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2432)] = condval_12;
  float condval_13;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_13 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 25056)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2560)] = condval_13;
  PadInput_shared[(((int)threadIdx.x) + 2688)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 2688) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_14;
  if (((1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_14 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 2816) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2816)] = condval_14;
  PadInput_shared[(((int)threadIdx.x) + 2944)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 2944) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_15;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_15 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 3072) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_15 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3072)] = condval_15;
  float condval_16;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_16 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 32224)];
  } else {
    condval_16 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3200)] = condval_16;
  PadInput_shared[(((int)threadIdx.x) + 3328)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 3328) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_17;
  if (((1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_17 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 3456) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_17 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3456)] = condval_17;
  PadInput_shared[(((int)threadIdx.x) + 3584)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 3584) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_18;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_18 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 3712) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_18 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3712)] = condval_18;
  float condval_19;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_19 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 39392)];
  } else {
    condval_19 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3840)] = condval_19;
  PadInput_shared[(((int)threadIdx.x) + 3968)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 3968) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_20;
  if (((1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_20 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 4096) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_20 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 4096)] = condval_20;
  PadInput_shared[(((int)threadIdx.x) + 4224)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 4224) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_21;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_21 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 4352) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_21 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 4352)] = condval_21;
  float condval_22;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_22 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 46560)];
  } else {
    condval_22 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 4480)] = condval_22;
  PadInput_shared[(((int)threadIdx.x) + 4608)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 4608) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_23;
  if (((1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_23 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 4736) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_23 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 4736)] = condval_23;
  PadInput_shared[(((int)threadIdx.x) + 4864)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 4864) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  float condval_24;
  if (((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107)) {
    condval_24 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 4992) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_24 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 4992)] = condval_24;
  float condval_25;
  if ((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)))) {
    condval_25 = placeholder[(((((((int)blockIdx.x) / 14) * 57344) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) + 53728)];
  } else {
    condval_25 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 5120)] = condval_25;
  PadInput_shared[(((int)threadIdx.x) + 5248)] = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 5248) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3488)];
  float condval_26;
  if (((((((((int)blockIdx.x) / 14) * 16) + ((((int)threadIdx.x) + 5376) / 320)) < 113) && (1 <= (((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + (((((int)threadIdx.x) >> 5) + 8) % 10)) < 113))) {
    condval_26 = placeholder[(((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 5376) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((((int)threadIdx.x) >> 5) + 8) % 10) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_26 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 5376)] = condval_26;
  float condval_27;
  if (((((((int)blockIdx.x) / 14) * 16) + ((((int)threadIdx.x) + 5504) / 320)) < 113)) {
    condval_27 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 5504) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3552)];
  } else {
    condval_27 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 5504)] = condval_27;
  float condval_28;
  if ((((((((int)blockIdx.x) / 14) * 16) + ((((int)threadIdx.x) + 5632) / 320)) < 113) && ((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) >> 5)) < 107))) {
    condval_28 = placeholder[((((((((int)blockIdx.x) / 14) * 57344) + (((((int)threadIdx.x) + 5632) / 320) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_28 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 5632)] = condval_28;
  placeholder_shared[((int)threadIdx.x)] = placeholder_1[((int)threadIdx.x)];
  placeholder_shared[(((((((int)threadIdx.x) + 128) / 96) * 96) + ((((((int)threadIdx.x) >> 5) + 1) % 3) * 32)) + (((int)threadIdx.x) & 31))] = placeholder_1[(((((((int)threadIdx.x) + 128) / 96) * 96) + ((((((int)threadIdx.x) >> 5) + 1) % 3) * 32)) + (((int)threadIdx.x) & 31))];
  if (((int)threadIdx.x) < 32) {
    placeholder_shared[(((int)threadIdx.x) + 256)] = placeholder_1[(((int)threadIdx.x) + 256)];
  }
  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31))] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 320)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 640)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 960)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1600)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1920)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2240)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 32)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 352)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1632)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 64)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 32)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 352)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1632)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 64)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 96)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 64)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 96)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 448)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 768)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1728)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 96)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 448)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 768)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1728)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 160)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 480)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 800)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1440)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1760)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2080)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2400)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 320)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 640)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 960)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1600)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1920)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2560)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 352)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1632)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2592)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 352)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1632)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2592)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2656)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2656)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 448)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 768)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1728)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2656)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 448)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 768)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1728)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 480)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 800)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1440)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1760)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2080)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2400)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2720)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 640)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 960)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1600)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1920)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2560)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2880)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1632)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2592)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2912)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2944)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 672)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1632)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2592)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2912)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2944)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2656)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2976)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 704)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1664)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2624)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2944)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2656)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2976)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 768)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1728)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 3008)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 736)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1696)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2656)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2976)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 768)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1728)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 3008)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 800)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1440)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 1760)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2080)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2400)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 2720)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 6) * 2560) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 3040)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 32)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 64)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 96)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 3584)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 3616)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 3648)] = depth_conv2d_nhwc_local[6];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 3680)] = depth_conv2d_nhwc_local[7];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 7168)] = depth_conv2d_nhwc_local[8];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 7200)] = depth_conv2d_nhwc_local[9];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 7232)] = depth_conv2d_nhwc_local[10];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 7264)] = depth_conv2d_nhwc_local[11];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 10752)] = depth_conv2d_nhwc_local[12];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 10784)] = depth_conv2d_nhwc_local[13];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 10816)] = depth_conv2d_nhwc_local[14];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 10848)] = depth_conv2d_nhwc_local[15];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 14336)] = depth_conv2d_nhwc_local[16];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 14368)] = depth_conv2d_nhwc_local[17];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 14400)] = depth_conv2d_nhwc_local[18];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 14432)] = depth_conv2d_nhwc_local[19];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 17920)] = depth_conv2d_nhwc_local[20];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 17952)] = depth_conv2d_nhwc_local[21];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 17984)] = depth_conv2d_nhwc_local[22];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 18016)] = depth_conv2d_nhwc_local[23];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 21504)] = depth_conv2d_nhwc_local[24];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 21536)] = depth_conv2d_nhwc_local[25];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 21568)] = depth_conv2d_nhwc_local[26];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 21600)] = depth_conv2d_nhwc_local[27];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 25088)] = depth_conv2d_nhwc_local[28];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 25120)] = depth_conv2d_nhwc_local[29];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 25152)] = depth_conv2d_nhwc_local[30];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 14) * 57344) + ((((int)threadIdx.x) >> 6) * 28672)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (((int)threadIdx.x) & 31)) + 25184)] = depth_conv2d_nhwc_local[31];
}

