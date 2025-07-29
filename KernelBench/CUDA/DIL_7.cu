
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
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[5];
  __shared__ float PadInput_shared[2880];
  __shared__ float weight_shared[576];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 64; ++rc_0) {
    __syncthreads();
    float condval;
    if ((((40 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval = inputs[(((((((((int)threadIdx.x) / 40) * 3584) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[((int)threadIdx.x)] = condval;
    float condval_1;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_1 = inputs[((((((((((int)threadIdx.x) + 64) / 40) * 3584) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 64) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_1;
    float condval_2;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_2 = inputs[((((((((((int)threadIdx.x) + 128) / 40) * 3584) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 128) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_2;
    float condval_3;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_3 = inputs[((((((((((int)threadIdx.x) + 192) / 40) * 3584) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 192) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_3;
    float condval_4;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_4 = inputs[((((((((((int)threadIdx.x) + 256) / 40) * 3584) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 256) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_4;
    float condval_5;
    if (((((1 <= (((((int)threadIdx.x) / 40) + 8) % 9)) && (((((int)threadIdx.x) + 320) % 360) < 320)) && (1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_5 = inputs[(((((((((((int)threadIdx.x) + 320) / 360) * 25088) + ((((((int)threadIdx.x) / 40) + 8) % 9) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 320)] = condval_5;
    float condval_6;
    if ((((16 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_6 = inputs[(((((((((((int)threadIdx.x) + 384) / 360) * 25088) + (((((int)threadIdx.x) + 24) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 384) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_6;
    float condval_7;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_7 = inputs[(((((((((((int)threadIdx.x) + 448) / 360) * 25088) + (((((int)threadIdx.x) + 88) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 448) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_7;
    float condval_8;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_8 = inputs[(((((((((((int)threadIdx.x) + 512) / 360) * 25088) + (((((int)threadIdx.x) + 152) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 512) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_8;
    float condval_9;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_9 = inputs[(((((((((((int)threadIdx.x) + 576) / 360) * 25088) + (((((int)threadIdx.x) + 216) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 576) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_9;
    float condval_10;
    if ((((((int)threadIdx.x) < 40) && (1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_10 = inputs[((((((((((int)threadIdx.x) + 640) / 360) * 25088) + ((((int)blockIdx.x) >> 6) * 512)) + ((((int)threadIdx.x) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 20992)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 640)] = condval_10;
    float condval_11;
    if (((((5 <= (((((int)threadIdx.x) >> 3) + 43) % 45)) && (((((int)threadIdx.x) + 344) % 360) < 320)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_11 = inputs[(((((((((((int)threadIdx.x) + 704) / 360) * 25088) + (((((((int)threadIdx.x) >> 3) + 43) % 45) / 5) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_11 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 704) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_11;
    float condval_12;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_12 = inputs[(((((((((((int)threadIdx.x) + 768) / 360) * 25088) + (((((int)threadIdx.x) + 48) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_12 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 768) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_12;
    float condval_13;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_13 = inputs[(((((((((((int)threadIdx.x) + 832) / 360) * 25088) + (((((int)threadIdx.x) + 112) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_13 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 832) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_13;
    float condval_14;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_14 = inputs[(((((((((((int)threadIdx.x) + 896) / 360) * 25088) + (((((int)threadIdx.x) + 176) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_14 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 896) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_14;
    float condval_15;
    if (((1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_15 = inputs[(((((((((((int)threadIdx.x) + 960) / 360) * 25088) + ((((int)threadIdx.x) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 17408)];
    } else {
      condval_15 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 960)] = condval_15;
    float condval_16;
    if (((((5 <= (((((int)threadIdx.x) >> 3) + 38) % 45)) && (((((int)threadIdx.x) + 304) % 360) < 320)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_16 = inputs[(((((((((((int)threadIdx.x) + 1024) / 360) * 25088) + (((((((int)threadIdx.x) >> 3) + 38) % 45) / 5) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_16 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1024) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_16;
    float condval_17;
    if ((((32 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_17 = inputs[(((((((((((int)threadIdx.x) + 1088) / 360) * 25088) + (((((int)threadIdx.x) + 8) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_17 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1088) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_17;
    float condval_18;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_18 = inputs[(((((((((((int)threadIdx.x) + 1152) / 360) * 25088) + (((((int)threadIdx.x) + 72) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_18 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1152) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_18;
    float condval_19;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_19 = inputs[(((((((((((int)threadIdx.x) + 1216) / 360) * 25088) + (((((int)threadIdx.x) + 136) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_19 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1216) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_19;
    float condval_20;
    if (((1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_20 = inputs[(((((((((((int)threadIdx.x) + 1280) / 360) * 25088) + ((((int)threadIdx.x) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 13824)];
    } else {
      condval_20 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1280)] = condval_20;
    float condval_21;
    if ((((((int)threadIdx.x) < 56) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_21 = inputs[(((((((((((int)threadIdx.x) + 1344) / 360) * 25088) + (((((int)threadIdx.x) + 264) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_21 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1344) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_21;
    float condval_22;
    if (((((5 <= (((((int)threadIdx.x) >> 3) + 41) % 45)) && (((((int)threadIdx.x) + 328) % 360) < 320)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_22 = inputs[(((((((((((int)threadIdx.x) + 1408) / 360) * 25088) + (((((((int)threadIdx.x) >> 3) + 41) % 45) / 5) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_22 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1408) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_22;
    float condval_23;
    if ((((8 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_23 = inputs[(((((((((((int)threadIdx.x) + 1472) / 360) * 25088) + (((((int)threadIdx.x) + 32) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_23 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1472) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_23;
    float condval_24;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_24 = inputs[(((((((((((int)threadIdx.x) + 1536) / 360) * 25088) + (((((int)threadIdx.x) + 96) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_24 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1536) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_24;
    float condval_25;
    if (((1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_25 = inputs[(((((((((((int)threadIdx.x) + 1600) / 360) * 25088) + ((((int)threadIdx.x) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 10240)];
    } else {
      condval_25 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1600)] = condval_25;
    float condval_26;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_26 = inputs[(((((((((((int)threadIdx.x) + 1664) / 360) * 25088) + (((((int)threadIdx.x) + 224) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_26 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1664) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_26;
    float condval_27;
    if ((((((int)threadIdx.x) < 32) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_27 = inputs[(((((((((((int)threadIdx.x) + 1728) / 360) * 25088) + (((((int)threadIdx.x) + 288) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((int)threadIdx.x) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3584)];
    } else {
      condval_27 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1728) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_27;
    float condval_28;
    if (((((5 <= (((((int)threadIdx.x) >> 3) + 44) % 45)) && (((((int)threadIdx.x) + 352) % 360) < 320)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_28 = inputs[(((((((((((int)threadIdx.x) + 1792) / 360) * 25088) + (((((((int)threadIdx.x) >> 3) + 44) % 45) / 5) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_28 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1792) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_28;
    float condval_29;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_29 = inputs[(((((((((((int)threadIdx.x) + 1856) / 360) * 25088) + (((((int)threadIdx.x) + 56) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_29 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1856) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_29;
    float condval_30;
    if (((1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_30 = inputs[(((((((((((int)threadIdx.x) + 1920) / 360) * 25088) + ((((int)threadIdx.x) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 6656)];
    } else {
      condval_30 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1920)] = condval_30;
    float condval_31;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_31 = inputs[(((((((((((int)threadIdx.x) + 1984) / 360) * 25088) + (((((int)threadIdx.x) + 184) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_31 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 1984) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_31;
    float condval_32;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_32 = inputs[(((((((((((int)threadIdx.x) + 2048) / 360) * 25088) + (((((int)threadIdx.x) + 248) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_32 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2048) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_32;
    float condval_33;
    if (((((5 <= (((((int)threadIdx.x) >> 3) + 39) % 45)) && (((((int)threadIdx.x) + 312) % 360) < 320)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_33 = inputs[(((((((((((int)threadIdx.x) + 2112) / 360) * 25088) + (((((((int)threadIdx.x) >> 3) + 39) % 45) / 5) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_33 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2112) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_33;
    float condval_34;
    if ((((24 <= ((int)threadIdx.x)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_34 = inputs[(((((((((((int)threadIdx.x) + 2176) / 360) * 25088) + (((((int)threadIdx.x) + 16) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_34 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2176) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_34;
    float condval_35;
    if (((1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_35 = inputs[(((((((((((int)threadIdx.x) + 2240) / 360) * 25088) + ((((int)threadIdx.x) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 3072)];
    } else {
      condval_35 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2240)] = condval_35;
    float condval_36;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_36 = inputs[(((((((((((int)threadIdx.x) + 2304) / 360) * 25088) + (((((int)threadIdx.x) + 144) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_36 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2304) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_36;
    float condval_37;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_37 = inputs[(((((((((((int)threadIdx.x) + 2368) / 360) * 25088) + (((((int)threadIdx.x) + 208) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_37 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2368) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_37;
    float condval_38;
    if ((((((int)threadIdx.x) < 48) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_38 = inputs[(((((((((((int)threadIdx.x) + 2432) / 360) * 25088) + (((((int)threadIdx.x) + 272) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_38 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2432) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_38;
    float condval_39;
    if (((((5 <= (((((int)threadIdx.x) >> 3) + 42) % 45)) && (((((int)threadIdx.x) + 336) % 360) < 320)) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_39 = inputs[(((((((((((int)threadIdx.x) + 2496) / 360) * 25088) + (((((((int)threadIdx.x) >> 3) + 42) % 45) / 5) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_39 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2496) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_39;
    float condval_40;
    if (((1 <= ((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3))) && (((((int)blockIdx.x) >> 6) + ((((int)threadIdx.x) % 40) >> 3)) < 8))) {
      condval_40 = inputs[(((((((((((int)threadIdx.x) + 2560) / 360) * 25088) + ((((int)threadIdx.x) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) % 40) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 512)];
    } else {
      condval_40 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2560)] = condval_40;
    float condval_41;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 3) % 5)) < 8))) {
      condval_41 = inputs[(((((((((((int)threadIdx.x) + 2624) / 360) * 25088) + (((((int)threadIdx.x) + 104) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_41 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2624) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 3) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_41;
    float condval_42;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 1) % 5)) < 8))) {
      condval_42 = inputs[(((((((((((int)threadIdx.x) + 2688) / 360) * 25088) + (((((int)threadIdx.x) + 168) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_42 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2688) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 1) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_42;
    float condval_43;
    if (((1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 4) % 5)) < 8))) {
      condval_43 = inputs[(((((((((((int)threadIdx.x) + 2752) / 360) * 25088) + (((((int)threadIdx.x) + 232) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
    } else {
      condval_43 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2752) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 4) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_43;
    float condval_44;
    if ((((((int)threadIdx.x) < 24) && (1 <= ((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)))) && (((((int)blockIdx.x) >> 6) + (((((int)threadIdx.x) >> 3) + 2) % 5)) < 8))) {
      condval_44 = inputs[(((((((((((int)threadIdx.x) + 2816) / 360) * 25088) + (((((int)threadIdx.x) + 296) / 40) * 3584)) + ((((int)blockIdx.x) >> 6) * 512)) + ((((int)threadIdx.x) >> 3) * 512)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3072)];
    } else {
      condval_44 = 0.000000e+00f;
    }
    PadInput_shared[(((((((int)threadIdx.x) + 2816) / 40) * 40) + ((((((int)threadIdx.x) >> 3) + 2) % 5) * 8)) + (((int)threadIdx.x) & 7))] = condval_44;
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((((int)threadIdx.x) >> 4) * 262144) + (rc_0 * 4096)) + (((((int)threadIdx.x) & 15) >> 1) * 512)) + ((((int)blockIdx.x) & 63) * 8)) + ((((int)threadIdx.x) & 1) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(weight + (((((((((int)threadIdx.x) >> 4) * 262144) + (rc_0 * 4096)) + (((((int)threadIdx.x) & 15) >> 1) * 512)) + ((((int)blockIdx.x) & 63) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1048576));
    if (((int)threadIdx.x) < 16) {
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(weight + (((((rc_0 * 4096) + ((((int)threadIdx.x) >> 1) * 512)) + ((((int)blockIdx.x) & 63) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 2097152));
    }
    __syncthreads();
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) >> 3) * 360)] * weight_shared[(((int)threadIdx.x) & 7)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 1)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 2)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 3)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 40)] * weight_shared[(((int)threadIdx.x) & 7)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 41)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 42)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 43)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 80)] * weight_shared[(((int)threadIdx.x) & 7)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 81)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 82)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 83)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 120)] * weight_shared[(((int)threadIdx.x) & 7)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 121)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 122)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 123)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 160)] * weight_shared[(((int)threadIdx.x) & 7)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 161)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 162)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 163)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 4)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 5)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 6)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 7)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 44)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 45)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 46)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 47)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 84)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 85)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 86)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 87)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 124)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 125)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 126)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 127)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 164)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 165)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 166)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 167)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 16)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 17)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 18)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 19)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 56)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 57)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 58)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 59)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 96)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 97)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 98)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 99)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 136)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 137)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 138)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 139)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 176)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 177)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 178)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 179)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 20)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 21)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 22)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 23)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 60)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 61)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 62)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 63)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 100)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 101)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 102)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 103)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 140)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 141)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 142)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 143)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 180)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 181)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 182)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 183)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 32)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 33)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 34)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 35)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 72)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 73)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 74)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 75)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 112)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 113)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 114)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 115)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 152)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 153)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 154)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 155)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 192)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 193)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 194)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 195)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 36)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 37)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 38)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 39)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 76)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 77)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 78)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 79)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 116)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 117)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 118)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 119)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 156)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 157)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 158)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 159)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 196)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 197)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 198)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 199)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 80)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 81)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 82)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 83)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 120)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 121)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 122)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 123)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 160)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 161)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 162)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 163)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 200)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 201)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 202)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 203)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 240)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 241)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 242)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 243)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 84)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 85)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 86)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 87)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 124)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 125)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 126)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 127)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 164)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 165)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 166)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 167)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 204)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 205)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 206)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 207)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 244)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 245)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 246)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 247)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 96)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 97)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 98)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 99)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 136)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 137)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 138)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 139)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 176)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 177)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 178)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 179)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 216)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 217)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 218)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 219)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 256)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 257)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 258)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 259)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 100)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 101)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 102)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 103)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 140)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 141)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 142)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 143)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 180)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 181)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 182)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 183)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 220)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 221)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 222)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 223)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 260)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 261)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 262)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 263)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 112)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 113)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 114)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 115)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 152)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 153)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 154)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 155)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 192)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 193)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 194)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 195)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 232)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 233)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 234)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 235)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 272)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 273)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 274)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 275)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 116)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 117)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 118)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 119)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 156)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 157)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 158)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 159)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 196)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 197)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 198)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 199)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 236)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 237)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 238)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 239)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 276)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 277)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 278)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 279)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 160)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 161)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 162)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 163)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 200)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 201)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 202)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 203)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 240)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 241)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 242)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 243)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 280)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 281)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 282)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 283)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 320)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 321)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 322)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 323)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 164)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 165)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 166)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 167)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 204)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 205)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 206)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 207)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 244)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 245)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 246)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 247)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 284)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 285)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 286)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 287)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 324)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 325)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 326)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 327)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 176)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 177)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 178)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 179)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 216)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 217)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 218)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 219)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 256)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 257)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 258)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 259)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 296)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 297)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 298)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 299)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 336)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 337)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 338)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 339)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 180)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 181)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 182)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 183)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 220)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 221)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 222)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 223)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 260)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 261)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 262)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 263)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 300)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 301)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 302)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 303)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 340)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 341)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 342)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 343)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 192)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 193)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 194)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 195)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 232)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 233)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 234)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 235)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 272)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 273)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 274)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 275)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 312)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 313)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 314)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 315)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 352)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 353)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 354)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 355)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 196)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 197)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 198)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 199)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 236)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 237)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 238)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 239)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 276)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 277)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 278)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 279)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 316)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 317)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 318)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 319)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 356)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 357)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 358)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 3) * 360) + 359)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
  }
  conv2d_nhwc[((((((int)threadIdx.x) >> 3) * 12800) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((int)threadIdx.x) >> 3) * 12800) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) & 7)) + 2560)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((int)threadIdx.x) >> 3) * 12800) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) & 7)) + 5120)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((int)threadIdx.x) >> 3) * 12800) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) & 7)) + 7680)] = conv2d_nhwc_local[3];
  conv2d_nhwc[(((((((int)threadIdx.x) >> 3) * 12800) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) & 7)) + 10240)] = conv2d_nhwc_local[4];
}

