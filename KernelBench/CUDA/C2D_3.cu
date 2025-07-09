
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
extern "C" __global__ void __launch_bounds__(56) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(56) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[2];
  __shared__ float PadInput_shared[1728];
  __shared__ float weight_shared[9216];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  float condval;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
    condval = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((int)threadIdx.x) & 15) * 3) >> 4) * 512)) + ((((int)threadIdx.x) * 3) & 15)) - 4096)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) * 3)] = condval;
  float condval_1;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
    condval_1 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 1) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 4096)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 1)] = condval_1;
  float condval_2;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
    condval_2 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 2) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 4096)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 2)] = condval_2;
  float condval_3;
  if (((1 <= ((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_3 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 8) & 15)) - 4096)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 168)] = condval_3;
  float condval_4;
  if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_4 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 9) & 15)) - 4096)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 169)] = condval_4;
  float condval_5;
  if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_5 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 10) & 15)) - 4096)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 170)] = condval_5;
  if (((int)threadIdx.x) < 32) {
    float condval_6;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
      condval_6 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + (((((int)threadIdx.x) * 3) >> 4) * 512)) + ((((int)threadIdx.x) * 3) & 15)) + 20992)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 336)] = condval_6;
    float condval_7;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
      condval_7 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 20992)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 337)] = condval_7;
    float condval_8;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
      condval_8 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 20992)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 338)] = condval_8;
  }
  *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((int)threadIdx.x) >> 2) * 512) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 224)) = *(float4*)(weight + ((((((((int)threadIdx.x) + 56) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 448)) = *(float4*)(weight + ((((((((int)threadIdx.x) + 112) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 12) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 672)) = *(float4*)(weight + ((((((((int)threadIdx.x) + 168) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 10) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(weight + ((((((((int)threadIdx.x) + 224) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 8) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1120)) = *(float4*)(weight + ((((((((int)threadIdx.x) + 280) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 6) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1344)) = *(float4*)(weight + ((((((((int)threadIdx.x) + 336) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 4) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1568)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 392) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1024));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(weight + (((((((int)threadIdx.x) >> 2) * 512) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1835008));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2016)) = *(float4*)(weight + ((((((((int)threadIdx.x) + 504) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  if (((int)threadIdx.x) < 16) {
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2240)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 560) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 6144));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float condval_9;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
    condval_9 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((int)threadIdx.x) & 15) * 3) >> 4) * 512)) + ((((int)threadIdx.x) * 3) & 15)) - 4080)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 432)] = condval_9;
  float condval_10;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
    condval_10 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 1) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 4080)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 433)] = condval_10;
  float condval_11;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
    condval_11 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 2) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 4080)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 434)] = condval_11;
  float condval_12;
  if (((1 <= ((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_12 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 8) & 15)) - 4080)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 600)] = condval_12;
  float condval_13;
  if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_13 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 9) & 15)) - 4080)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 601)] = condval_13;
  float condval_14;
  if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_14 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 10) & 15)) - 4080)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 602)] = condval_14;
  if (((int)threadIdx.x) < 32) {
    float condval_15;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
      condval_15 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + (((((int)threadIdx.x) * 3) >> 4) * 512)) + ((((int)threadIdx.x) * 3) & 15)) + 21008)];
    } else {
      condval_15 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 768)] = condval_15;
    float condval_16;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
      condval_16 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 21008)];
    } else {
      condval_16 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 769)] = condval_16;
    float condval_17;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
      condval_17 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 21008)];
    } else {
      condval_17 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 770)] = condval_17;
  }
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(weight + (((((((int)threadIdx.x) >> 2) * 512) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2528)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 56) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2752)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 112) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 12) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2976)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 168) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 10) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3200)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 224) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 8) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3424)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 280) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 6) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3648)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 336) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 4) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3872)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 392) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 9216));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(float4*)(weight + (((((((int)threadIdx.x) >> 2) * 512) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1843200));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4320)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 504) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 8192));
  if (((int)threadIdx.x) < 16) {
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4544)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 560) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 14336));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float condval_18;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
    condval_18 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((int)threadIdx.x) & 15) * 3) >> 4) * 512)) + ((((int)threadIdx.x) * 3) & 15)) - 4064)];
  } else {
    condval_18 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 864)] = condval_18;
  float condval_19;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
    condval_19 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 1) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 4064)];
  } else {
    condval_19 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 865)] = condval_19;
  float condval_20;
  if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
    condval_20 = inputs[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 2) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 4064)];
  } else {
    condval_20 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 866)] = condval_20;
  float condval_21;
  if (((1 <= ((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_21 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 8) & 15)) - 4064)];
  } else {
    condval_21 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 1032)] = condval_21;
  float condval_22;
  if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_22 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 9) & 15)) - 4064)];
  } else {
    condval_22 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 1033)] = condval_22;
  float condval_23;
  if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1)) < 8))) {
    condval_23 = inputs[(((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1) * 512)) + (((((int)threadIdx.x) * 3) + 10) & 15)) - 4064)];
  } else {
    condval_23 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 1034)] = condval_23;
  if (((int)threadIdx.x) < 32) {
    float condval_24;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
      condval_24 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + (((((int)threadIdx.x) * 3) >> 4) * 512)) + ((((int)threadIdx.x) * 3) & 15)) + 21024)];
    } else {
      condval_24 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 1200)] = condval_24;
    float condval_25;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
      condval_25 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 21024)];
    } else {
      condval_25 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 1201)] = condval_25;
    float condval_26;
    if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
      condval_26 = inputs[(((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 512)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 21024)];
    } else {
      condval_26 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 1202)] = condval_26;
  }
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4608)) = *(float4*)(weight + (((((((int)threadIdx.x) >> 2) * 512) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4832)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 56) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5056)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 112) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 12) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5280)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 168) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 10) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5504)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 224) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 8) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5728)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 280) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 6) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5952)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 336) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 4) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6176)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 392) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 17408));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6400)) = *(float4*)(weight + (((((((int)threadIdx.x) >> 2) * 512) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1851392));
  *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6624)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 504) >> 6) * 262144) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
  if (((int)threadIdx.x) < 16) {
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6848)) = *(float4*)(weight + (((((((((int)threadIdx.x) + 560) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 22528));
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int rh_0_rw_0_rc_0_fused = 0; rh_0_rw_0_rc_0_fused < 29; ++rh_0_rw_0_rc_0_fused) {
    __syncthreads();
    float condval_27;
    if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
      condval_27 = inputs[(((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((int)threadIdx.x) & 15) * 3) >> 4) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + ((((int)threadIdx.x) * 3) & 15)) - 4048)];
    } else {
      condval_27 = 0.000000e+00f;
    }
    PadInput_shared[((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3))] = condval_27;
    float condval_28;
    if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
      condval_28 = inputs[(((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 1) >> 4) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 4048)];
    } else {
      condval_28 = 0.000000e+00f;
    }
    PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 1)] = condval_28;
    float condval_29;
    if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
      condval_29 = inputs[(((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((int)threadIdx.x) & 15) * 3) + 2) >> 4) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 4048)];
    } else {
      condval_29 = 0.000000e+00f;
    }
    PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 2)] = condval_29;
    float condval_30;
    if (((1 <= ((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + (((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1)) < 8))) {
      condval_30 = inputs[((((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + ((((((((int)threadIdx.x) * 3) >> 3) + 3) % 6) >> 1) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + (((((int)threadIdx.x) * 3) + 8) & 15)) - 4048)];
    } else {
      condval_30 = 0.000000e+00f;
    }
    PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 168)] = condval_30;
    float condval_31;
    if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1)) < 8))) {
      condval_31 = inputs[((((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 1) >> 3) + 3) % 6) >> 1) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + (((((int)threadIdx.x) * 3) + 9) & 15)) - 4048)];
    } else {
      condval_31 = 0.000000e+00f;
    }
    PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 169)] = condval_31;
    float condval_32;
    if (((1 <= ((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1))) && (((((int)blockIdx.x) >> 5) + ((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1)) < 8))) {
      condval_32 = inputs[((((((((((int)threadIdx.x) + 56) >> 4) * 3584) + ((((int)blockIdx.x) >> 5) * 512)) + (((((((((int)threadIdx.x) * 3) + 2) >> 3) + 3) % 6) >> 1) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + (((((int)threadIdx.x) * 3) + 10) & 15)) - 4048)];
    } else {
      condval_32 = 0.000000e+00f;
    }
    PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 170)] = condval_32;
    if (((int)threadIdx.x) < 32) {
      float condval_33;
      if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)))) && (((((int)blockIdx.x) >> 5) + (((((int)threadIdx.x) & 15) * 3) >> 4)) < 8))) {
        condval_33 = inputs[((((((((int)blockIdx.x) >> 5) * 512) + (((((int)threadIdx.x) * 3) >> 4) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 21040)];
      } else {
        condval_33 = 0.000000e+00f;
      }
      PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 336)] = condval_33;
      float condval_34;
      if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 1) >> 4)) < 8))) {
        condval_34 = inputs[((((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 21040)];
      } else {
        condval_34 = 0.000000e+00f;
      }
      PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 337)] = condval_34;
      float condval_35;
      if ((((((int)threadIdx.x) < 16) && (1 <= ((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)))) && (((((int)blockIdx.x) >> 5) + ((((((int)threadIdx.x) & 15) * 3) + 2) >> 4)) < 8))) {
        condval_35 = inputs[((((((((int)blockIdx.x) >> 5) * 512) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 512)) + (rh_0_rw_0_rc_0_fused * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 21040)];
      } else {
        condval_35 = 0.000000e+00f;
      }
      PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 432) + (((int)threadIdx.x) * 3)) + 338)] = condval_35;
    }
    *(float4*)(weight_shared + ((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4))) = *(float4*)(weight + (((((rh_0_rw_0_rc_0_fused * 8192) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 224)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 56) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 448)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 112) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((((int)threadIdx.x) >> 2) + 12) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 672)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 168) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((((int)threadIdx.x) >> 2) + 10) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 896)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 224) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((((int)threadIdx.x) >> 2) + 8) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 1120)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 280) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((((int)threadIdx.x) >> 2) + 6) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 1344)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 336) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((((int)threadIdx.x) >> 2) + 4) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 1568)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 392) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 25600));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 1792)) = *(float4*)(weight + (((((rh_0_rw_0_rc_0_fused * 8192) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1859584));
    *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 2016)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 504) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((((int)threadIdx.x) >> 2) + 14) & 15) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 24576));
    if (((int)threadIdx.x) < 16) {
      *(float4*)(weight_shared + (((((rh_0_rw_0_rc_0_fused + 3) & 3) * 2304) + (((int)threadIdx.x) * 4)) + 2240)) = *(float4*)(weight + ((((((((((int)threadIdx.x) + 560) >> 6) * 262144) + (rh_0_rw_0_rc_0_fused * 8192)) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 30720));
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

    __syncthreads();
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48))] * weight_shared[(((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2))]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 1)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 16)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 2)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 32)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 3)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 48)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 4)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 64)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 5)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 80)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 6)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 96)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 7)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 112)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 8)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 128)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 9)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 144)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 10)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 160)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 11)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 176)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 12)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 192)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 13)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 208)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 14)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 224)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 15)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 240)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 16)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 256)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 17)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 272)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 18)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 288)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 19)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 304)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 20)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 320)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 21)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 336)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 22)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 352)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 23)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 368)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 24)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 384)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 25)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 400)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 26)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 416)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 27)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 432)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 28)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 448)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 29)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 464)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 30)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 480)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 31)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 496)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 32)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 512)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 33)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 528)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 34)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 544)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 35)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 560)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 36)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 576)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 37)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 592)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 38)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 608)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 39)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 624)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 40)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 640)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 41)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 656)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 42)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 672)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 43)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 688)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 44)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 704)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 45)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 720)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 46)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 736)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 47)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 752)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48))] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 1)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 17)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 2)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 33)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 3)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 49)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 4)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 65)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 5)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 81)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 6)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 97)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 7)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 113)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 8)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 129)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 9)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 145)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 10)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 161)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 11)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 177)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 12)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 193)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 13)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 209)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 14)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 225)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 15)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 241)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 16)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 257)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 17)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 273)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 18)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 289)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 19)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 305)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 20)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 321)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 21)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 337)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 22)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 353)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 23)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 369)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 24)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 385)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 25)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 401)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 26)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 417)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 27)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 433)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 28)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 449)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 29)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 465)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 30)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 481)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 31)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 497)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 32)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 513)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 33)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 529)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 34)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 545)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 35)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 561)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 36)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 577)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 37)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 593)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 38)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 609)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 39)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 625)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 40)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 641)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 41)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 657)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 42)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 673)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 43)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 689)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 44)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 705)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 45)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 721)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 46)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 737)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 47)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 753)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 48)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 768)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 49)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 784)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 50)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 800)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 51)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 816)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 52)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 832)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 53)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 848)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 54)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 864)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 55)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 880)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 56)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 896)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 57)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 912)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 58)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 928)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 59)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 944)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 60)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 960)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 61)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 976)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 62)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 992)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 63)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1008)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 64)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1024)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 65)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1040)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 66)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1056)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 67)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1072)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 68)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1088)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 69)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1104)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 70)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1120)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 71)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1136)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 72)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1152)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 73)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1168)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 74)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1184)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 75)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1200)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 76)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1216)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 77)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1232)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 78)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1248)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 79)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1264)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 80)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1280)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 81)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1296)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 82)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1312)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 83)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1328)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 84)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1344)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 85)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1360)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 86)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1376)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 87)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1392)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 88)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1408)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 89)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1424)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 90)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1440)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 91)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1456)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 92)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1472)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 93)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1488)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 94)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1504)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 95)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1520)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 48)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 769)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 49)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 785)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 50)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 801)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 51)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 817)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 52)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 833)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 53)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 849)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 54)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 865)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 55)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 881)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 56)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 897)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 57)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 913)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 58)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 929)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 59)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 945)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 60)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 961)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 61)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 977)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 62)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 993)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 63)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1009)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 64)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1025)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 65)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1041)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 66)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1057)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 67)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1073)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 68)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1089)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 69)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1105)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 70)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1121)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 71)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1137)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 72)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1153)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 73)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1169)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 74)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1185)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 75)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1201)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 76)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1217)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 77)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1233)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 78)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1249)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 79)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1265)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 80)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1281)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 81)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1297)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 82)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1313)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 83)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1329)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 84)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1345)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 85)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1361)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 86)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1377)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 87)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1393)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 88)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1409)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 89)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1425)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 90)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1441)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 91)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1457)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 92)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1473)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 93)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1489)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 94)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1505)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 95)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1521)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 96)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1536)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 97)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1552)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 98)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1568)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 99)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1584)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 100)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1600)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 101)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1616)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 102)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1632)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 103)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1648)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 104)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1664)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 105)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1680)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 106)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1696)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 107)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1712)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 108)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1728)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 109)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1744)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 110)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1760)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 111)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1776)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 112)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1792)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 113)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1808)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 114)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1824)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 115)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1840)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 116)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1856)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 117)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1872)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 118)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1888)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 119)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1904)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 120)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1920)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 121)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1936)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 122)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1952)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 123)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1968)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 124)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1984)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 125)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2000)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 126)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2016)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 127)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2032)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 128)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2048)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 129)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2064)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 130)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2080)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 131)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2096)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 132)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2112)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 133)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2128)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 134)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2144)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 135)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2160)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 136)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2176)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 137)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2192)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 138)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2208)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 139)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2224)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 140)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2240)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 141)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2256)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 142)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2272)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 143)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2288)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 96)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1537)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 97)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1553)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 98)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1569)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 99)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1585)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 100)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1601)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 101)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1617)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 102)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1633)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 103)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1649)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 104)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1665)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 105)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1681)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 106)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1697)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 107)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1713)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 108)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1729)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 109)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1745)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 110)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1761)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 111)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1777)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 112)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1793)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 113)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1809)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 114)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1825)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 115)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1841)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 116)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1857)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 117)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1873)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 118)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1889)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 119)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1905)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 120)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1921)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 121)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1937)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 122)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1953)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 123)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1969)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 124)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 1985)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 125)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2001)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 126)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2017)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 127)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2033)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 128)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2049)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 129)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2065)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 130)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2081)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 131)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2097)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 132)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2113)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 133)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2129)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 134)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2145)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 135)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2161)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 136)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2177)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 137)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2193)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 138)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2209)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 139)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2225)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 140)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2241)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 141)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2257)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 142)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2273)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused & 3) * 432) + ((((int)threadIdx.x) >> 3) * 48)) + 143)] * weight_shared[((((rh_0_rw_0_rc_0_fused & 3) * 2304) + ((((int)threadIdx.x) & 7) * 2)) + 2289)]));
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 432)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2304)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 433)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2320)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 434)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2336)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 435)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2352)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 436)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2368)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 437)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2384)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 438)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2400)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 439)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2416)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 440)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2432)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 441)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2448)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 442)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2464)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 443)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2480)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 444)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2496)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 445)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2512)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 446)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2528)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 447)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2544)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 448)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2560)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 449)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2576)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 450)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2592)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 451)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2608)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 452)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2624)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 453)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2640)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 454)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2656)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 455)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2672)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 456)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2688)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 457)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2704)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 458)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2720)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 459)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2736)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 460)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2752)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 461)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2768)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 462)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2784)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 463)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2800)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 464)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2816)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 465)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2832)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 466)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2848)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 467)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2864)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 468)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2880)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 469)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2896)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 470)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2912)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 471)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2928)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 472)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2944)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 473)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2960)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 474)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2976)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 475)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2992)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 476)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3008)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 477)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3024)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 478)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3040)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 479)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3056)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 432)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2305)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 433)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2321)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 434)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2337)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 435)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2353)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 436)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2369)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 437)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2385)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 438)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2401)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 439)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2417)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 440)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2433)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 441)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2449)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 442)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2465)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 443)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2481)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 444)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2497)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 445)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2513)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 446)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2529)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 447)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2545)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 448)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2561)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 449)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2577)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 450)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2593)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 451)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2609)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 452)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2625)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 453)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2641)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 454)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2657)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 455)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2673)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 456)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2689)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 457)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2705)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 458)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2721)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 459)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2737)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 460)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2753)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 461)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2769)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 462)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2785)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 463)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2801)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 464)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2817)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 465)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2833)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 466)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2849)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 467)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2865)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 468)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2881)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 469)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2897)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 470)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2913)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 471)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2929)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 472)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2945)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 473)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2961)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 474)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2977)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 475)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 2993)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 476)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3009)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 477)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3025)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 478)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3041)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 479)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3057)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 480)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3072)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 481)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3088)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 482)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3104)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 483)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3120)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 484)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3136)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 485)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3152)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 486)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3168)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 487)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3184)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 488)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3200)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 489)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3216)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 490)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3232)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 491)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3248)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 492)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3264)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 493)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3280)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 494)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3296)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 495)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3312)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 496)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3328)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 497)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3344)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 498)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3360)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 499)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3376)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 500)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3392)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 501)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3408)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 502)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3424)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 503)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3440)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 504)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3456)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 505)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3472)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 506)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3488)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 507)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3504)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 508)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3520)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 509)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3536)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 510)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3552)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 511)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3568)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 512)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3584)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 513)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3600)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 514)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3616)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 515)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3632)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 516)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3648)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 517)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3664)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 518)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3680)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 519)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3696)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 520)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3712)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 521)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3728)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 522)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3744)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 523)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3760)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 524)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3776)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 525)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3792)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 526)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3808)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 527)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3824)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 480)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3073)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 481)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3089)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 482)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3105)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 483)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3121)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 484)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3137)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 485)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3153)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 486)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3169)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 487)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3185)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 488)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3201)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 489)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3217)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 490)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3233)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 491)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3249)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 492)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3265)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 493)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3281)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 494)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3297)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 495)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3313)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 496)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3329)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 497)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3345)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 498)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3361)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 499)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3377)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 500)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3393)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 501)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3409)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 502)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3425)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 503)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3441)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 504)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3457)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 505)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3473)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 506)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3489)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 507)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3505)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 508)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3521)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 509)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3537)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 510)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3553)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 511)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3569)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 512)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3585)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 513)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3601)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 514)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3617)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 515)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3633)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 516)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3649)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 517)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3665)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 518)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3681)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 519)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3697)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 520)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3713)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 521)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3729)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 522)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3745)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 523)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3761)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 524)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3777)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 525)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3793)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 526)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3809)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 527)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3825)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 528)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3840)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 529)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3856)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 530)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3872)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 531)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3888)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 532)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3904)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 533)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3920)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 534)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3936)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 535)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3952)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 536)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3968)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 537)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3984)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 538)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4000)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 539)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4016)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 540)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4032)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 541)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4048)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 542)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4064)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 543)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4080)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 544)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4096)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 545)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4112)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 546)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4128)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 547)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4144)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 548)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4160)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 549)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4176)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 550)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4192)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 551)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4208)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 552)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4224)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 553)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4240)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 554)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4256)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 555)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4272)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 556)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4288)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 557)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4304)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 558)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4320)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 559)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4336)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 560)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4352)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 561)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4368)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 562)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4384)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 563)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4400)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 564)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4416)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 565)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4432)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 566)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4448)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 567)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4464)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 568)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4480)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 569)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4496)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 570)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4512)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 571)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4528)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 572)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4544)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 573)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4560)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 574)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4576)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 575)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4592)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 528)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3841)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 529)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3857)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 530)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3873)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 531)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3889)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 532)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3905)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 533)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3921)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 534)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3937)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 535)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3953)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 536)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3969)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 537)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 3985)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 538)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4001)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 539)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4017)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 540)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4033)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 541)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4049)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 542)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4065)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 543)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4081)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 544)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4097)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 545)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4113)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 546)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4129)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 547)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4145)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 548)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4161)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 549)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4177)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 550)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4193)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 551)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4209)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 552)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4225)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 553)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4241)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 554)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4257)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 555)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4273)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 556)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4289)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 557)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4305)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 558)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4321)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 559)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4337)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 560)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4353)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 561)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4369)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 562)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4385)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 563)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4401)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 564)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4417)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 565)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4433)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 566)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4449)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 567)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4465)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 568)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4481)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 569)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4497)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 570)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4513)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 571)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4529)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 572)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4545)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 573)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4561)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 574)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4577)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 575)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4593)]));
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 864)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4608)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 865)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4624)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 866)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4640)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 867)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4656)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 868)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4672)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 869)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4688)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 870)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4704)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 871)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4720)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 872)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4736)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 873)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4752)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 874)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4768)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 875)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4784)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 876)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4800)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 877)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4816)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 878)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4832)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 879)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4848)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 880)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4864)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 881)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4880)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 882)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4896)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 883)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4912)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 884)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4928)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 885)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4944)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 886)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4960)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 887)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4976)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 888)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4992)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 889)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5008)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 890)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5024)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 891)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5040)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 892)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5056)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 893)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5072)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 894)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5088)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 895)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5104)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 896)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5120)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 897)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5136)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 898)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5152)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 899)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5168)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 900)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5184)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 901)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5200)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 902)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5216)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 903)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5232)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 904)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5248)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 905)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5264)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 906)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5280)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 907)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5296)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 908)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5312)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 909)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5328)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 910)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5344)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 911)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5360)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 864)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4609)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 865)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4625)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 866)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4641)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 867)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4657)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 868)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4673)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 869)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4689)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 870)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4705)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 871)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4721)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 872)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4737)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 873)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4753)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 874)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4769)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 875)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4785)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 876)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4801)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 877)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4817)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 878)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4833)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 879)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4849)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 880)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4865)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 881)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4881)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 882)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4897)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 883)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4913)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 884)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4929)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 885)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4945)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 886)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4961)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 887)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4977)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 888)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 4993)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 889)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5009)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 890)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5025)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 891)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5041)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 892)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5057)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 893)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5073)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 894)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5089)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 895)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5105)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 896)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5121)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 897)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5137)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 898)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5153)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 899)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5169)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 900)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5185)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 901)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5201)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 902)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5217)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 903)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5233)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 904)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5249)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 905)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5265)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 906)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5281)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 907)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5297)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 908)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5313)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 909)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5329)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 910)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5345)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 911)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5361)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 912)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5376)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 913)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5392)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 914)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5408)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 915)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5424)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 916)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5440)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 917)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5456)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 918)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5472)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 919)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5488)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 920)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5504)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 921)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5520)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 922)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5536)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 923)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5552)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 924)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5568)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 925)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5584)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 926)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5600)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 927)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5616)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 928)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5632)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 929)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5648)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 930)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5664)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 931)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5680)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 932)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5696)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 933)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5712)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 934)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5728)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 935)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5744)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 936)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5760)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 937)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5776)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 938)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5792)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 939)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5808)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 940)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5824)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 941)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5840)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 942)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5856)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 943)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5872)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 944)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5888)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 945)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5904)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 946)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5920)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 947)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5936)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 948)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5952)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 949)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5968)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 950)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5984)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 951)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6000)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 952)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6016)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 953)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6032)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 954)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6048)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 955)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6064)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 956)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6080)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 957)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6096)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 958)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6112)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 959)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6128)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 912)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5377)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 913)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5393)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 914)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5409)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 915)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5425)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 916)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5441)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 917)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5457)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 918)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5473)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 919)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5489)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 920)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5505)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 921)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5521)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 922)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5537)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 923)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5553)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 924)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5569)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 925)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5585)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 926)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5601)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 927)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5617)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 928)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5633)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 929)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5649)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 930)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5665)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 931)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5681)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 932)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5697)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 933)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5713)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 934)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5729)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 935)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5745)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 936)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5761)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 937)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5777)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 938)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5793)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 939)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5809)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 940)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5825)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 941)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5841)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 942)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5857)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 943)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5873)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 944)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5889)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 945)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5905)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 946)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5921)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 947)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5937)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 948)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5953)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 949)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5969)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 950)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 5985)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 951)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6001)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 952)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6017)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 953)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6033)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 954)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6049)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 955)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6065)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 956)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6081)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 957)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6097)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 958)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6113)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 959)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6129)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 960)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6144)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 961)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6160)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 962)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6176)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 963)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6192)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 964)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6208)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 965)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6224)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 966)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6240)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 967)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6256)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 968)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6272)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 969)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6288)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 970)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6304)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 971)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6320)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 972)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6336)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 973)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6352)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 974)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6368)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 975)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6384)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 976)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6400)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 977)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6416)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 978)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6432)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 979)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6448)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 980)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6464)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 981)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6480)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 982)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6496)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 983)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6512)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 984)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6528)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 985)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6544)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 986)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6560)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 987)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6576)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 988)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6592)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 989)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6608)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 990)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6624)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 991)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6640)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 992)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6656)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 993)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6672)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 994)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6688)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 995)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6704)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 996)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6720)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 997)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6736)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 998)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6752)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 999)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6768)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1000)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6784)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1001)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6800)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1002)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6816)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1003)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6832)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1004)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6848)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1005)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6864)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1006)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6880)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1007)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6896)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 960)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6145)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 961)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6161)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 962)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6177)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 963)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6193)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 964)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6209)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 965)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6225)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 966)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6241)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 967)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6257)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 968)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6273)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 969)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6289)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 970)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6305)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 971)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6321)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 972)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6337)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 973)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6353)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 974)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6369)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 975)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6385)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 976)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6401)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 977)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6417)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 978)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6433)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 979)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6449)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 980)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6465)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 981)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6481)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 982)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6497)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 983)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6513)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 984)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6529)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 985)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6545)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 986)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6561)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 987)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6577)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 988)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6593)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 989)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6609)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 990)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6625)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 991)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6641)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 992)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6657)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 993)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6673)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 994)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6689)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 995)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6705)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 996)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6721)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 997)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6737)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 998)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6753)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 999)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6769)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1000)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6785)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1001)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6801)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1002)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6817)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1003)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6833)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1004)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6849)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1005)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6865)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1006)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6881)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1007)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6897)]));
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1296)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6912)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1297)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6928)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1298)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6944)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1299)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6960)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1300)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6976)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1301)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6992)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1302)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7008)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1303)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7024)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1304)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7040)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1305)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7056)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1306)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7072)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1307)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7088)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1308)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7104)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1309)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7120)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1310)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7136)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1311)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7152)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1312)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7168)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1313)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7184)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1314)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7200)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1315)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7216)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1316)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7232)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1317)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7248)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1318)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7264)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1319)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7280)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1320)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7296)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1321)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7312)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1322)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7328)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1323)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7344)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1324)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7360)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1325)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7376)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1326)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7392)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1327)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7408)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1328)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7424)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1329)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7440)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1330)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7456)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1331)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7472)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1332)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7488)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1333)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7504)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1334)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7520)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1335)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7536)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1336)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7552)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1337)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7568)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1338)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7584)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1339)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7600)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1340)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7616)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1341)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7632)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1342)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7648)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1343)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7664)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1296)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6913)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1297)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6929)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1298)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6945)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1299)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6961)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1300)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6977)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1301)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 6993)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1302)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7009)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1303)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7025)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1304)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7041)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1305)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7057)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1306)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7073)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1307)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7089)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1308)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7105)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1309)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7121)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1310)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7137)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1311)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7153)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1312)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7169)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1313)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7185)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1314)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7201)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1315)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7217)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1316)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7233)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1317)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7249)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1318)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7265)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1319)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7281)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1320)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7297)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1321)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7313)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1322)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7329)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1323)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7345)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1324)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7361)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1325)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7377)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1326)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7393)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1327)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7409)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1328)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7425)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1329)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7441)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1330)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7457)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1331)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7473)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1332)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7489)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1333)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7505)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1334)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7521)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1335)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7537)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1336)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7553)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1337)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7569)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1338)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7585)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1339)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7601)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1340)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7617)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1341)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7633)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1342)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7649)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1343)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7665)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1344)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7680)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1345)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7696)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1346)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7712)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1347)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7728)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1348)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7744)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1349)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7760)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1350)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7776)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1351)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7792)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1352)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7808)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1353)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7824)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1354)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7840)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1355)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7856)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1356)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7872)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1357)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7888)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1358)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7904)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1359)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7920)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1360)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7936)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1361)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7952)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1362)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7968)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1363)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7984)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1364)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8000)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1365)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8016)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1366)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8032)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1367)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8048)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1368)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8064)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1369)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8080)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1370)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8096)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1371)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8112)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1372)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8128)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1373)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8144)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1374)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8160)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1375)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8176)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1376)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8192)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1377)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8208)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1378)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8224)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1379)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8240)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1380)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8256)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1381)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8272)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1382)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8288)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1383)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8304)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1384)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8320)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1385)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8336)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1386)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8352)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1387)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8368)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1388)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8384)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1389)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8400)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1390)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8416)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1391)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8432)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1344)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7681)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1345)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7697)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1346)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7713)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1347)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7729)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1348)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7745)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1349)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7761)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1350)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7777)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1351)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7793)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1352)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7809)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1353)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7825)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1354)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7841)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1355)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7857)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1356)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7873)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1357)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7889)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1358)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7905)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1359)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7921)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1360)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7937)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1361)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7953)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1362)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7969)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1363)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 7985)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1364)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8001)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1365)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8017)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1366)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8033)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1367)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8049)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1368)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8065)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1369)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8081)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1370)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8097)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1371)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8113)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1372)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8129)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1373)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8145)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1374)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8161)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1375)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8177)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1376)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8193)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1377)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8209)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1378)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8225)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1379)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8241)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1380)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8257)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1381)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8273)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1382)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8289)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1383)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8305)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1384)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8321)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1385)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8337)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1386)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8353)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1387)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8369)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1388)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8385)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1389)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8401)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1390)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8417)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1391)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8433)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1392)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8448)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1393)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8464)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1394)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8480)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1395)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8496)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1396)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8512)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1397)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8528)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1398)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8544)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1399)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8560)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1400)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8576)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1401)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8592)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1402)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8608)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1403)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8624)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1404)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8640)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1405)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8656)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1406)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8672)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1407)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8688)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1408)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8704)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1409)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8720)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1410)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8736)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1411)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8752)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1412)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8768)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1413)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8784)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1414)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8800)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1415)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8816)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1416)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8832)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1417)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8848)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1418)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8864)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1419)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8880)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1420)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8896)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1421)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8912)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1422)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8928)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1423)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8944)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1424)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8960)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1425)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8976)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1426)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8992)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1427)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9008)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1428)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9024)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1429)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9040)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1430)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9056)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1431)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9072)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1432)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9088)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1433)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9104)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1434)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9120)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1435)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9136)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1436)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9152)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1437)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9168)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1438)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9184)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1439)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9200)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1392)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8449)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1393)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8465)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1394)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8481)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1395)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8497)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1396)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8513)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1397)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8529)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1398)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8545)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1399)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8561)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1400)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8577)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1401)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8593)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1402)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8609)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1403)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8625)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1404)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8641)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1405)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8657)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1406)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8673)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1407)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8689)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1408)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8705)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1409)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8721)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1410)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8737)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1411)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8753)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1412)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8769)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1413)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8785)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1414)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8801)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1415)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8817)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1416)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8833)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1417)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8849)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1418)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8865)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1419)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8881)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1420)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8897)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1421)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8913)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1422)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8929)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1423)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8945)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1424)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8961)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1425)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8977)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1426)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 8993)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1427)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9009)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1428)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9025)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1429)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9041)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1430)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9057)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1431)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9073)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1432)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9089)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1433)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9105)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1434)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9121)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1435)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9137)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1436)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9153)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1437)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9169)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1438)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9185)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 48) + 1439)] * weight_shared[(((((int)threadIdx.x) & 7) * 2) + 9201)]));
  conv2d_nhwc[((((((int)threadIdx.x) >> 3) * 3584) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((int)threadIdx.x) >> 3) * 3584) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1)] = conv2d_nhwc_local[1];
}

