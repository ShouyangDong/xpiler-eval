
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
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[32];
  __shared__ float PadInput_shared[8064];
  __shared__ float placeholder_shared[96];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  depth_conv2d_nhwc_local[7] = 0.000000e+00f;
  depth_conv2d_nhwc_local[8] = 0.000000e+00f;
  depth_conv2d_nhwc_local[9] = 0.000000e+00f;
  depth_conv2d_nhwc_local[10] = 0.000000e+00f;
  depth_conv2d_nhwc_local[11] = 0.000000e+00f;
  depth_conv2d_nhwc_local[12] = 0.000000e+00f;
  depth_conv2d_nhwc_local[13] = 0.000000e+00f;
  depth_conv2d_nhwc_local[14] = 0.000000e+00f;
  depth_conv2d_nhwc_local[15] = 0.000000e+00f;
  depth_conv2d_nhwc_local[16] = 0.000000e+00f;
  depth_conv2d_nhwc_local[17] = 0.000000e+00f;
  depth_conv2d_nhwc_local[18] = 0.000000e+00f;
  depth_conv2d_nhwc_local[19] = 0.000000e+00f;
  depth_conv2d_nhwc_local[20] = 0.000000e+00f;
  depth_conv2d_nhwc_local[21] = 0.000000e+00f;
  depth_conv2d_nhwc_local[22] = 0.000000e+00f;
  depth_conv2d_nhwc_local[23] = 0.000000e+00f;
  depth_conv2d_nhwc_local[24] = 0.000000e+00f;
  depth_conv2d_nhwc_local[25] = 0.000000e+00f;
  depth_conv2d_nhwc_local[26] = 0.000000e+00f;
  depth_conv2d_nhwc_local[27] = 0.000000e+00f;
  depth_conv2d_nhwc_local[28] = 0.000000e+00f;
  depth_conv2d_nhwc_local[29] = 0.000000e+00f;
  depth_conv2d_nhwc_local[30] = 0.000000e+00f;
  depth_conv2d_nhwc_local[31] = 0.000000e+00f;
  for (int rw_0 = 0; rw_0 < 3; ++rw_0) {
    __syncthreads();
    float condval;
    if ((((16 <= (((int)blockIdx.x) % 112)) && (1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) - 3616)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[((int)threadIdx.x)] = condval;
    float condval_1;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_1 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) - 32)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 224)] = condval_1;
    float condval_2;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_2 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 3552)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 448)] = condval_2;
    float condval_3;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_3 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 7136)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 672)] = condval_3;
    float condval_4;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_4 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 10720)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 896)] = condval_4;
    float condval_5;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_5 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 14304)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1120)] = condval_5;
    float condval_6;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_6 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 17888)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1344)] = condval_6;
    float condval_7;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_7 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 21472)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1568)] = condval_7;
    float condval_8;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_8 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 25056)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1792)] = condval_8;
    float condval_9;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_9 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 28640)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2016)] = condval_9;
    float condval_10;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_10 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 32224)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2240)] = condval_10;
    float condval_11;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_11 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 35808)];
    } else {
      condval_11 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2464)] = condval_11;
    float condval_12;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_12 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 39392)];
    } else {
      condval_12 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2688)] = condval_12;
    float condval_13;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_13 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 42976)];
    } else {
      condval_13 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2912)] = condval_13;
    float condval_14;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_14 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 46560)];
    } else {
      condval_14 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 3136)] = condval_14;
    float condval_15;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_15 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 50144)];
    } else {
      condval_15 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 3360)] = condval_15;
    float condval_16;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_16 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 53728)];
    } else {
      condval_16 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 3584)] = condval_16;
    float condval_17;
    if (((((((int)blockIdx.x) % 112) < 96) && (1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_17 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 57312)];
    } else {
      condval_17 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 3808)] = condval_17;
    float condval_18;
    if ((((16 <= (((int)blockIdx.x) % 112)) && (1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_18 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 397792)];
    } else {
      condval_18 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 4032)] = condval_18;
    float condval_19;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_19 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 401376)];
    } else {
      condval_19 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 4256)] = condval_19;
    float condval_20;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_20 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 404960)];
    } else {
      condval_20 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 4480)] = condval_20;
    float condval_21;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_21 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 408544)];
    } else {
      condval_21 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 4704)] = condval_21;
    float condval_22;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_22 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 412128)];
    } else {
      condval_22 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 4928)] = condval_22;
    float condval_23;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_23 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 415712)];
    } else {
      condval_23 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 5152)] = condval_23;
    float condval_24;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_24 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 419296)];
    } else {
      condval_24 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 5376)] = condval_24;
    float condval_25;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_25 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 422880)];
    } else {
      condval_25 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 5600)] = condval_25;
    float condval_26;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_26 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 426464)];
    } else {
      condval_26 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 5824)] = condval_26;
    float condval_27;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_27 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 430048)];
    } else {
      condval_27 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 6048)] = condval_27;
    float condval_28;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_28 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 433632)];
    } else {
      condval_28 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 6272)] = condval_28;
    float condval_29;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_29 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 437216)];
    } else {
      condval_29 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 6496)] = condval_29;
    float condval_30;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_30 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 440800)];
    } else {
      condval_30 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 6720)] = condval_30;
    float condval_31;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_31 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 444384)];
    } else {
      condval_31 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 6944)] = condval_31;
    float condval_32;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_32 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 447968)];
    } else {
      condval_32 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 7168)] = condval_32;
    float condval_33;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_33 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 451552)];
    } else {
      condval_33 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 7392)] = condval_33;
    float condval_34;
    if (((1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0)) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_34 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 455136)];
    } else {
      condval_34 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 7616)] = condval_34;
    float condval_35;
    if (((((((int)blockIdx.x) % 112) < 96) && (1 <= ((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 7) + (((int)threadIdx.x) >> 5)) + rw_0) < 113))) {
      condval_35 = placeholder[(((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + (rw_0 * 32)) + ((int)threadIdx.x)) + 458720)];
    } else {
      condval_35 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 7840)] = condval_35;
    if (((int)threadIdx.x) < 24) {
      *(float4*)(placeholder_shared + (((int)threadIdx.x) * 4)) = *(float4*)(placeholder_1 + ((((((int)threadIdx.x) >> 3) * 96) + (rw_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    }
    __syncthreads();
    for (int rh_1 = 0; rh_1 < 3; ++rh_1) {
      depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((rh_1 * 224) + ((int)threadIdx.x))] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 224)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 448)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 672)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 896)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 1120)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 1344)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 1568)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 1792)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 2016)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 2240)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 2464)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 2688)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 2912)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 3136)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 3360)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[16] = (depth_conv2d_nhwc_local[16] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 4032)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[17] = (depth_conv2d_nhwc_local[17] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 4256)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[18] = (depth_conv2d_nhwc_local[18] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 4480)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[19] = (depth_conv2d_nhwc_local[19] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 4704)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[20] = (depth_conv2d_nhwc_local[20] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 4928)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[21] = (depth_conv2d_nhwc_local[21] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 5152)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[22] = (depth_conv2d_nhwc_local[22] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 5376)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[23] = (depth_conv2d_nhwc_local[23] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 5600)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[24] = (depth_conv2d_nhwc_local[24] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 5824)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[25] = (depth_conv2d_nhwc_local[25] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 6048)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[26] = (depth_conv2d_nhwc_local[26] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 6272)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[27] = (depth_conv2d_nhwc_local[27] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 6496)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[28] = (depth_conv2d_nhwc_local[28] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 6720)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[29] = (depth_conv2d_nhwc_local[29] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 6944)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[30] = (depth_conv2d_nhwc_local[30] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 7168)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
      depth_conv2d_nhwc_local[31] = (depth_conv2d_nhwc_local[31] + (PadInput_shared[(((rh_1 * 224) + ((int)threadIdx.x)) + 7392)] * placeholder_shared[((rh_1 * 32) + (((int)threadIdx.x) & 31))]));
    }
  }
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 3584)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 7168)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 10752)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 14336)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 17920)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 21504)] = depth_conv2d_nhwc_local[6];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 25088)] = depth_conv2d_nhwc_local[7];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 28672)] = depth_conv2d_nhwc_local[8];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 32256)] = depth_conv2d_nhwc_local[9];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 35840)] = depth_conv2d_nhwc_local[10];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 39424)] = depth_conv2d_nhwc_local[11];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 43008)] = depth_conv2d_nhwc_local[12];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 46592)] = depth_conv2d_nhwc_local[13];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 50176)] = depth_conv2d_nhwc_local[14];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 53760)] = depth_conv2d_nhwc_local[15];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 401408)] = depth_conv2d_nhwc_local[16];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 404992)] = depth_conv2d_nhwc_local[17];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 408576)] = depth_conv2d_nhwc_local[18];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 412160)] = depth_conv2d_nhwc_local[19];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 415744)] = depth_conv2d_nhwc_local[20];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 419328)] = depth_conv2d_nhwc_local[21];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 422912)] = depth_conv2d_nhwc_local[22];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 426496)] = depth_conv2d_nhwc_local[23];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 430080)] = depth_conv2d_nhwc_local[24];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 433664)] = depth_conv2d_nhwc_local[25];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 437248)] = depth_conv2d_nhwc_local[26];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 440832)] = depth_conv2d_nhwc_local[27];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 444416)] = depth_conv2d_nhwc_local[28];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 448000)] = depth_conv2d_nhwc_local[29];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 451584)] = depth_conv2d_nhwc_local[30];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 802816) + (((((int)blockIdx.x) % 112) >> 4) * 57344)) + ((((int)blockIdx.x) & 15) * 224)) + ((int)threadIdx.x)) + 455168)] = depth_conv2d_nhwc_local[31];
}

