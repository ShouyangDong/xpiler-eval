
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ conv2d_capsule_nhwijc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ conv2d_capsule_nhwijc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_capsule_nhwijc_local[2];
  __shared__ float PadInput_shared[864];
  __shared__ float weight_shared[1152];
  conv2d_capsule_nhwijc_local[0] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[1] = 0.000000e+00f;
  float condval;
  if ((((((int)blockIdx.x) >> 7) == 1) && (1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5))))) {
    condval = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) - 2176)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) * 3)] = condval;
  float condval_1;
  if ((((((int)blockIdx.x) >> 7) == 1) && (1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5))))) {
    condval_1 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 2176)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 1)] = condval_1;
  float condval_2;
  if ((((((int)blockIdx.x) >> 7) == 1) && (1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5))))) {
    condval_2 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 2176)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 2)] = condval_2;
  float condval_3;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_3 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) - 128)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 96)] = condval_3;
  float condval_4;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_4 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 128)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 97)] = condval_4;
  float condval_5;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_5 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 128)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 98)] = condval_5;
  float condval_6;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_6 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 1920)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 192)] = condval_6;
  float condval_7;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_7 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 1920)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 193)] = condval_7;
  float condval_8;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_8 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 1920)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 194)] = condval_8;
  float condval_9;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_9 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 3968)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 288)] = condval_9;
  float condval_10;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_10 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 3968)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 289)] = condval_10;
  float condval_11;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_11 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 3968)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 290)] = condval_11;
  float condval_12;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_12 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 6016)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 384)] = condval_12;
  float condval_13;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_13 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 6016)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 385)] = condval_13;
  float condval_14;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_14 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 6016)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 386)] = condval_14;
  float condval_15;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_15 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 8064)];
  } else {
    condval_15 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 480)] = condval_15;
  float condval_16;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_16 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 8064)];
  } else {
    condval_16 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 481)] = condval_16;
  float condval_17;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_17 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 8064)];
  } else {
    condval_17 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 482)] = condval_17;
  float condval_18;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_18 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 10112)];
  } else {
    condval_18 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 576)] = condval_18;
  float condval_19;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_19 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 10112)];
  } else {
    condval_19 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 577)] = condval_19;
  float condval_20;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_20 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 10112)];
  } else {
    condval_20 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 578)] = condval_20;
  float condval_21;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_21 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 12160)];
  } else {
    condval_21 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 672)] = condval_21;
  float condval_22;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_22 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 12160)];
  } else {
    condval_22 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 673)] = condval_22;
  float condval_23;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_23 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 12160)];
  } else {
    condval_23 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 674)] = condval_23;
  float condval_24;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_24 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 14208)];
  } else {
    condval_24 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 768)] = condval_24;
  float condval_25;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_25 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 14208)];
  } else {
    condval_25 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 769)] = condval_25;
  float condval_26;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_26 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 14208)];
  } else {
    condval_26 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 770)] = condval_26;
  weight_shared[((int)threadIdx.x)] = weight[((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7))];
  weight_shared[(((int)threadIdx.x) + 32)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 64)];
  weight_shared[(((int)threadIdx.x) + 64)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 512)];
  weight_shared[(((int)threadIdx.x) + 96)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 576)];
  weight_shared[(((int)threadIdx.x) + 128)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 2048)];
  weight_shared[(((int)threadIdx.x) + 160)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 2112)];
  weight_shared[(((int)threadIdx.x) + 192)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 2560)];
  weight_shared[(((int)threadIdx.x) + 224)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 2624)];
  weight_shared[(((int)threadIdx.x) + 256)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 4096)];
  weight_shared[(((int)threadIdx.x) + 288)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 4160)];
  weight_shared[(((int)threadIdx.x) + 320)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 4608)];
  weight_shared[(((int)threadIdx.x) + 352)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 4672)];
  weight_shared[(((int)threadIdx.x) + 384)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 6144)];
  weight_shared[(((int)threadIdx.x) + 416)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 6208)];
  weight_shared[(((int)threadIdx.x) + 448)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 6656)];
  weight_shared[(((int)threadIdx.x) + 480)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 6720)];
  weight_shared[(((int)threadIdx.x) + 512)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 8192)];
  weight_shared[(((int)threadIdx.x) + 544)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 8256)];
  weight_shared[(((int)threadIdx.x) + 576)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 8704)];
  weight_shared[(((int)threadIdx.x) + 608)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 8768)];
  weight_shared[(((int)threadIdx.x) + 640)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 10240)];
  weight_shared[(((int)threadIdx.x) + 672)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 10304)];
  weight_shared[(((int)threadIdx.x) + 704)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 10752)];
  weight_shared[(((int)threadIdx.x) + 736)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 10816)];
  weight_shared[(((int)threadIdx.x) + 768)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 12288)];
  weight_shared[(((int)threadIdx.x) + 800)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 12352)];
  weight_shared[(((int)threadIdx.x) + 832)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 12800)];
  weight_shared[(((int)threadIdx.x) + 864)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 12864)];
  weight_shared[(((int)threadIdx.x) + 896)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 14336)];
  weight_shared[(((int)threadIdx.x) + 928)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 14400)];
  weight_shared[(((int)threadIdx.x) + 960)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 14848)];
  weight_shared[(((int)threadIdx.x) + 992)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 14912)];
  weight_shared[(((int)threadIdx.x) + 1024)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 16384)];
  weight_shared[(((int)threadIdx.x) + 1056)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 16448)];
  weight_shared[(((int)threadIdx.x) + 1088)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 16896)];
  weight_shared[(((int)threadIdx.x) + 1120)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 16960)];
  __syncthreads();
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16))] * weight_shared[(((int)threadIdx.x) & 7)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 192)] * weight_shared[(((int)threadIdx.x) & 7)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 1)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 193)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 2)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 194)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 3)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 195)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 8)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 200)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 9)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 201)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 10)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 202)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 11)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 203)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 96)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 288)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 97)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 289)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 98)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 290)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 99)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 291)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 104)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 296)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 105)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 297)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 106)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 298)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 107)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 299)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 192)] * weight_shared[((((int)threadIdx.x) & 7) + 768)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 384)] * weight_shared[((((int)threadIdx.x) & 7) + 768)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 193)] * weight_shared[((((int)threadIdx.x) & 7) + 776)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 385)] * weight_shared[((((int)threadIdx.x) & 7) + 776)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 194)] * weight_shared[((((int)threadIdx.x) & 7) + 784)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 386)] * weight_shared[((((int)threadIdx.x) & 7) + 784)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 195)] * weight_shared[((((int)threadIdx.x) & 7) + 792)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 387)] * weight_shared[((((int)threadIdx.x) & 7) + 792)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 200)] * weight_shared[((((int)threadIdx.x) & 7) + 832)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 392)] * weight_shared[((((int)threadIdx.x) & 7) + 832)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 201)] * weight_shared[((((int)threadIdx.x) & 7) + 840)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 393)] * weight_shared[((((int)threadIdx.x) & 7) + 840)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 202)] * weight_shared[((((int)threadIdx.x) & 7) + 848)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 394)] * weight_shared[((((int)threadIdx.x) & 7) + 848)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 203)] * weight_shared[((((int)threadIdx.x) & 7) + 856)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 395)] * weight_shared[((((int)threadIdx.x) & 7) + 856)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 4)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 196)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 5)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 197)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 6)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 198)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 7)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 199)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 12)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 204)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 13)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 205)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 14)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 206)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 15)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 207)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 100)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 292)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 101)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 293)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 102)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 294)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 103)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 295)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 108)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 300)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 109)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 301)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 110)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 302)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 111)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 303)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 196)] * weight_shared[((((int)threadIdx.x) & 7) + 800)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 388)] * weight_shared[((((int)threadIdx.x) & 7) + 800)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 197)] * weight_shared[((((int)threadIdx.x) & 7) + 808)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 389)] * weight_shared[((((int)threadIdx.x) & 7) + 808)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 198)] * weight_shared[((((int)threadIdx.x) & 7) + 816)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 390)] * weight_shared[((((int)threadIdx.x) & 7) + 816)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 199)] * weight_shared[((((int)threadIdx.x) & 7) + 824)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 391)] * weight_shared[((((int)threadIdx.x) & 7) + 824)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 204)] * weight_shared[((((int)threadIdx.x) & 7) + 864)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 396)] * weight_shared[((((int)threadIdx.x) & 7) + 864)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 205)] * weight_shared[((((int)threadIdx.x) & 7) + 872)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 397)] * weight_shared[((((int)threadIdx.x) & 7) + 872)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 206)] * weight_shared[((((int)threadIdx.x) & 7) + 880)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 398)] * weight_shared[((((int)threadIdx.x) & 7) + 880)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 207)] * weight_shared[((((int)threadIdx.x) & 7) + 888)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 399)] * weight_shared[((((int)threadIdx.x) & 7) + 888)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 32)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 224)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 33)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 225)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 34)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 226)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 35)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 227)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 40)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 232)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 41)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 233)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 42)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 234)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 43)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 235)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 128)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 320)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 129)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 321)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 130)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 322)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 131)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 323)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 136)] * weight_shared[((((int)threadIdx.x) & 7) + 576)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 328)] * weight_shared[((((int)threadIdx.x) & 7) + 576)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 137)] * weight_shared[((((int)threadIdx.x) & 7) + 584)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 329)] * weight_shared[((((int)threadIdx.x) & 7) + 584)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 138)] * weight_shared[((((int)threadIdx.x) & 7) + 592)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 330)] * weight_shared[((((int)threadIdx.x) & 7) + 592)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 139)] * weight_shared[((((int)threadIdx.x) & 7) + 600)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 331)] * weight_shared[((((int)threadIdx.x) & 7) + 600)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 224)] * weight_shared[((((int)threadIdx.x) & 7) + 896)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 416)] * weight_shared[((((int)threadIdx.x) & 7) + 896)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 225)] * weight_shared[((((int)threadIdx.x) & 7) + 904)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 417)] * weight_shared[((((int)threadIdx.x) & 7) + 904)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 226)] * weight_shared[((((int)threadIdx.x) & 7) + 912)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 418)] * weight_shared[((((int)threadIdx.x) & 7) + 912)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 227)] * weight_shared[((((int)threadIdx.x) & 7) + 920)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 419)] * weight_shared[((((int)threadIdx.x) & 7) + 920)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 232)] * weight_shared[((((int)threadIdx.x) & 7) + 960)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 424)] * weight_shared[((((int)threadIdx.x) & 7) + 960)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 233)] * weight_shared[((((int)threadIdx.x) & 7) + 968)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 425)] * weight_shared[((((int)threadIdx.x) & 7) + 968)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 234)] * weight_shared[((((int)threadIdx.x) & 7) + 976)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 426)] * weight_shared[((((int)threadIdx.x) & 7) + 976)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 235)] * weight_shared[((((int)threadIdx.x) & 7) + 984)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 427)] * weight_shared[((((int)threadIdx.x) & 7) + 984)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 36)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 228)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 37)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 229)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 38)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 230)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 39)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 231)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 44)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 236)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 45)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 237)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 46)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 238)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 47)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 239)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 132)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 324)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 133)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 325)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 134)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 326)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 135)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 327)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 140)] * weight_shared[((((int)threadIdx.x) & 7) + 608)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 332)] * weight_shared[((((int)threadIdx.x) & 7) + 608)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 141)] * weight_shared[((((int)threadIdx.x) & 7) + 616)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 333)] * weight_shared[((((int)threadIdx.x) & 7) + 616)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 142)] * weight_shared[((((int)threadIdx.x) & 7) + 624)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 334)] * weight_shared[((((int)threadIdx.x) & 7) + 624)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 143)] * weight_shared[((((int)threadIdx.x) & 7) + 632)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 335)] * weight_shared[((((int)threadIdx.x) & 7) + 632)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 228)] * weight_shared[((((int)threadIdx.x) & 7) + 928)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 420)] * weight_shared[((((int)threadIdx.x) & 7) + 928)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 229)] * weight_shared[((((int)threadIdx.x) & 7) + 936)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 421)] * weight_shared[((((int)threadIdx.x) & 7) + 936)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 230)] * weight_shared[((((int)threadIdx.x) & 7) + 944)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 422)] * weight_shared[((((int)threadIdx.x) & 7) + 944)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 231)] * weight_shared[((((int)threadIdx.x) & 7) + 952)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 423)] * weight_shared[((((int)threadIdx.x) & 7) + 952)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 236)] * weight_shared[((((int)threadIdx.x) & 7) + 992)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 428)] * weight_shared[((((int)threadIdx.x) & 7) + 992)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 237)] * weight_shared[((((int)threadIdx.x) & 7) + 1000)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 429)] * weight_shared[((((int)threadIdx.x) & 7) + 1000)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 238)] * weight_shared[((((int)threadIdx.x) & 7) + 1008)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 430)] * weight_shared[((((int)threadIdx.x) & 7) + 1008)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 239)] * weight_shared[((((int)threadIdx.x) & 7) + 1016)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 431)] * weight_shared[((((int)threadIdx.x) & 7) + 1016)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 64)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 256)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 65)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 257)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 66)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 258)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 67)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 259)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 72)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 264)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 73)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 265)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 74)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 266)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 75)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 267)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 160)] * weight_shared[((((int)threadIdx.x) & 7) + 640)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 352)] * weight_shared[((((int)threadIdx.x) & 7) + 640)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 161)] * weight_shared[((((int)threadIdx.x) & 7) + 648)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 353)] * weight_shared[((((int)threadIdx.x) & 7) + 648)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 162)] * weight_shared[((((int)threadIdx.x) & 7) + 656)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 354)] * weight_shared[((((int)threadIdx.x) & 7) + 656)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 163)] * weight_shared[((((int)threadIdx.x) & 7) + 664)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 355)] * weight_shared[((((int)threadIdx.x) & 7) + 664)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 168)] * weight_shared[((((int)threadIdx.x) & 7) + 704)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 360)] * weight_shared[((((int)threadIdx.x) & 7) + 704)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 169)] * weight_shared[((((int)threadIdx.x) & 7) + 712)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 361)] * weight_shared[((((int)threadIdx.x) & 7) + 712)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 170)] * weight_shared[((((int)threadIdx.x) & 7) + 720)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 362)] * weight_shared[((((int)threadIdx.x) & 7) + 720)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 171)] * weight_shared[((((int)threadIdx.x) & 7) + 728)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 363)] * weight_shared[((((int)threadIdx.x) & 7) + 728)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 256)] * weight_shared[((((int)threadIdx.x) & 7) + 1024)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 448)] * weight_shared[((((int)threadIdx.x) & 7) + 1024)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 257)] * weight_shared[((((int)threadIdx.x) & 7) + 1032)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 449)] * weight_shared[((((int)threadIdx.x) & 7) + 1032)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 258)] * weight_shared[((((int)threadIdx.x) & 7) + 1040)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 450)] * weight_shared[((((int)threadIdx.x) & 7) + 1040)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 259)] * weight_shared[((((int)threadIdx.x) & 7) + 1048)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 451)] * weight_shared[((((int)threadIdx.x) & 7) + 1048)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 264)] * weight_shared[((((int)threadIdx.x) & 7) + 1088)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 456)] * weight_shared[((((int)threadIdx.x) & 7) + 1088)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 265)] * weight_shared[((((int)threadIdx.x) & 7) + 1096)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 457)] * weight_shared[((((int)threadIdx.x) & 7) + 1096)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 266)] * weight_shared[((((int)threadIdx.x) & 7) + 1104)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 458)] * weight_shared[((((int)threadIdx.x) & 7) + 1104)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 267)] * weight_shared[((((int)threadIdx.x) & 7) + 1112)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 459)] * weight_shared[((((int)threadIdx.x) & 7) + 1112)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 68)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 260)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 69)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 261)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 70)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 262)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 71)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 263)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 76)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 268)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 77)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 269)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 78)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 270)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 79)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 271)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 164)] * weight_shared[((((int)threadIdx.x) & 7) + 672)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 356)] * weight_shared[((((int)threadIdx.x) & 7) + 672)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 165)] * weight_shared[((((int)threadIdx.x) & 7) + 680)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 357)] * weight_shared[((((int)threadIdx.x) & 7) + 680)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 166)] * weight_shared[((((int)threadIdx.x) & 7) + 688)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 358)] * weight_shared[((((int)threadIdx.x) & 7) + 688)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 167)] * weight_shared[((((int)threadIdx.x) & 7) + 696)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 359)] * weight_shared[((((int)threadIdx.x) & 7) + 696)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 172)] * weight_shared[((((int)threadIdx.x) & 7) + 736)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 364)] * weight_shared[((((int)threadIdx.x) & 7) + 736)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 173)] * weight_shared[((((int)threadIdx.x) & 7) + 744)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 365)] * weight_shared[((((int)threadIdx.x) & 7) + 744)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 174)] * weight_shared[((((int)threadIdx.x) & 7) + 752)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 366)] * weight_shared[((((int)threadIdx.x) & 7) + 752)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 175)] * weight_shared[((((int)threadIdx.x) & 7) + 760)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 367)] * weight_shared[((((int)threadIdx.x) & 7) + 760)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 260)] * weight_shared[((((int)threadIdx.x) & 7) + 1056)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 452)] * weight_shared[((((int)threadIdx.x) & 7) + 1056)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 261)] * weight_shared[((((int)threadIdx.x) & 7) + 1064)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 453)] * weight_shared[((((int)threadIdx.x) & 7) + 1064)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 262)] * weight_shared[((((int)threadIdx.x) & 7) + 1072)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 454)] * weight_shared[((((int)threadIdx.x) & 7) + 1072)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 263)] * weight_shared[((((int)threadIdx.x) & 7) + 1080)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 455)] * weight_shared[((((int)threadIdx.x) & 7) + 1080)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 268)] * weight_shared[((((int)threadIdx.x) & 7) + 1120)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 460)] * weight_shared[((((int)threadIdx.x) & 7) + 1120)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 269)] * weight_shared[((((int)threadIdx.x) & 7) + 1128)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 461)] * weight_shared[((((int)threadIdx.x) & 7) + 1128)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 270)] * weight_shared[((((int)threadIdx.x) & 7) + 1136)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 462)] * weight_shared[((((int)threadIdx.x) & 7) + 1136)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 271)] * weight_shared[((((int)threadIdx.x) & 7) + 1144)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 463)] * weight_shared[((((int)threadIdx.x) & 7) + 1144)]));
  __syncthreads();
  float condval_27;
  if ((((((int)blockIdx.x) >> 7) == 1) && (1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5))))) {
    condval_27 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) - 2160)];
  } else {
    condval_27 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) * 3)] = condval_27;
  float condval_28;
  if ((((((int)blockIdx.x) >> 7) == 1) && (1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5))))) {
    condval_28 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 2160)];
  } else {
    condval_28 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 1)] = condval_28;
  float condval_29;
  if ((((((int)blockIdx.x) >> 7) == 1) && (1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5))))) {
    condval_29 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 2160)];
  } else {
    condval_29 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 2)] = condval_29;
  float condval_30;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_30 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) - 112)];
  } else {
    condval_30 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 96)] = condval_30;
  float condval_31;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_31 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) - 112)];
  } else {
    condval_31 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 97)] = condval_31;
  float condval_32;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_32 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) - 112)];
  } else {
    condval_32 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 98)] = condval_32;
  float condval_33;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_33 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 1936)];
  } else {
    condval_33 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 192)] = condval_33;
  float condval_34;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_34 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 1936)];
  } else {
    condval_34 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 193)] = condval_34;
  float condval_35;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_35 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 1936)];
  } else {
    condval_35 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 194)] = condval_35;
  float condval_36;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_36 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 3984)];
  } else {
    condval_36 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 288)] = condval_36;
  float condval_37;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_37 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 3984)];
  } else {
    condval_37 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 289)] = condval_37;
  float condval_38;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_38 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 3984)];
  } else {
    condval_38 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 290)] = condval_38;
  float condval_39;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_39 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 6032)];
  } else {
    condval_39 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 384)] = condval_39;
  float condval_40;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_40 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 6032)];
  } else {
    condval_40 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 385)] = condval_40;
  float condval_41;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_41 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 6032)];
  } else {
    condval_41 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 386)] = condval_41;
  float condval_42;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_42 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 8080)];
  } else {
    condval_42 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 480)] = condval_42;
  float condval_43;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_43 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 8080)];
  } else {
    condval_43 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 481)] = condval_43;
  float condval_44;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_44 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 8080)];
  } else {
    condval_44 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 482)] = condval_44;
  float condval_45;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_45 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 10128)];
  } else {
    condval_45 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 576)] = condval_45;
  float condval_46;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_46 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 10128)];
  } else {
    condval_46 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 577)] = condval_46;
  float condval_47;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_47 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 10128)];
  } else {
    condval_47 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 578)] = condval_47;
  float condval_48;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_48 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 12176)];
  } else {
    condval_48 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 672)] = condval_48;
  float condval_49;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_49 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 12176)];
  } else {
    condval_49 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 673)] = condval_49;
  float condval_50;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_50 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 12176)];
  } else {
    condval_50 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 674)] = condval_50;
  float condval_51;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + ((((int)threadIdx.x) * 3) >> 5)))) {
    condval_51 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + (((((int)threadIdx.x) * 3) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + ((((((int)threadIdx.x) * 3) & 31) >> 4) * 32)) + ((((int)threadIdx.x) * 3) & 15)) + 14224)];
  } else {
    condval_51 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 768)] = condval_51;
  float condval_52;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 5)))) {
    condval_52 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 1) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 14224)];
  } else {
    condval_52 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 769)] = condval_52;
  float condval_53;
  if ((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 5)))) {
    condval_53 = inputs[((((((((((int)blockIdx.x) >> 7) * 16384) + (((((int)blockIdx.x) & 127) >> 4) * 256)) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 128)) + (((((int)blockIdx.x) & 15) >> 3) * 64)) + (((((((int)threadIdx.x) * 3) + 2) & 31) >> 4) * 32)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 14224)];
  } else {
    condval_53 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 3) + 770)] = condval_53;
  weight_shared[((int)threadIdx.x)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 1024)];
  weight_shared[(((int)threadIdx.x) + 32)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 1088)];
  weight_shared[(((int)threadIdx.x) + 64)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 1536)];
  weight_shared[(((int)threadIdx.x) + 96)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 1600)];
  weight_shared[(((int)threadIdx.x) + 128)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 3072)];
  weight_shared[(((int)threadIdx.x) + 160)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 3136)];
  weight_shared[(((int)threadIdx.x) + 192)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 3584)];
  weight_shared[(((int)threadIdx.x) + 224)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 3648)];
  weight_shared[(((int)threadIdx.x) + 256)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 5120)];
  weight_shared[(((int)threadIdx.x) + 288)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 5184)];
  weight_shared[(((int)threadIdx.x) + 320)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 5632)];
  weight_shared[(((int)threadIdx.x) + 352)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 5696)];
  weight_shared[(((int)threadIdx.x) + 384)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 7168)];
  weight_shared[(((int)threadIdx.x) + 416)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 7232)];
  weight_shared[(((int)threadIdx.x) + 448)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 7680)];
  weight_shared[(((int)threadIdx.x) + 480)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 7744)];
  weight_shared[(((int)threadIdx.x) + 512)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 9216)];
  weight_shared[(((int)threadIdx.x) + 544)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 9280)];
  weight_shared[(((int)threadIdx.x) + 576)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 9728)];
  weight_shared[(((int)threadIdx.x) + 608)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 9792)];
  weight_shared[(((int)threadIdx.x) + 640)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 11264)];
  weight_shared[(((int)threadIdx.x) + 672)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 11328)];
  weight_shared[(((int)threadIdx.x) + 704)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 11776)];
  weight_shared[(((int)threadIdx.x) + 736)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 11840)];
  weight_shared[(((int)threadIdx.x) + 768)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 13312)];
  weight_shared[(((int)threadIdx.x) + 800)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 13376)];
  weight_shared[(((int)threadIdx.x) + 832)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 13824)];
  weight_shared[(((int)threadIdx.x) + 864)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 13888)];
  weight_shared[(((int)threadIdx.x) + 896)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 15360)];
  weight_shared[(((int)threadIdx.x) + 928)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 15424)];
  weight_shared[(((int)threadIdx.x) + 960)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 15872)];
  weight_shared[(((int)threadIdx.x) + 992)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 15936)];
  weight_shared[(((int)threadIdx.x) + 1024)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 17408)];
  weight_shared[(((int)threadIdx.x) + 1056)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 17472)];
  weight_shared[(((int)threadIdx.x) + 1088)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 17920)];
  weight_shared[(((int)threadIdx.x) + 1120)] = weight[(((((((((int)blockIdx.x) & 7) >> 1) * 128) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 17984)];
  __syncthreads();
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16))] * weight_shared[(((int)threadIdx.x) & 7)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 192)] * weight_shared[(((int)threadIdx.x) & 7)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 1)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 193)] * weight_shared[((((int)threadIdx.x) & 7) + 8)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 2)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 194)] * weight_shared[((((int)threadIdx.x) & 7) + 16)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 3)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 195)] * weight_shared[((((int)threadIdx.x) & 7) + 24)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 8)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 200)] * weight_shared[((((int)threadIdx.x) & 7) + 64)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 9)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 201)] * weight_shared[((((int)threadIdx.x) & 7) + 72)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 10)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 202)] * weight_shared[((((int)threadIdx.x) & 7) + 80)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 11)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 203)] * weight_shared[((((int)threadIdx.x) & 7) + 88)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 96)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 288)] * weight_shared[((((int)threadIdx.x) & 7) + 384)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 97)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 289)] * weight_shared[((((int)threadIdx.x) & 7) + 392)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 98)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 290)] * weight_shared[((((int)threadIdx.x) & 7) + 400)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 99)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 291)] * weight_shared[((((int)threadIdx.x) & 7) + 408)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 104)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 296)] * weight_shared[((((int)threadIdx.x) & 7) + 448)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 105)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 297)] * weight_shared[((((int)threadIdx.x) & 7) + 456)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 106)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 298)] * weight_shared[((((int)threadIdx.x) & 7) + 464)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 107)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 299)] * weight_shared[((((int)threadIdx.x) & 7) + 472)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 192)] * weight_shared[((((int)threadIdx.x) & 7) + 768)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 384)] * weight_shared[((((int)threadIdx.x) & 7) + 768)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 193)] * weight_shared[((((int)threadIdx.x) & 7) + 776)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 385)] * weight_shared[((((int)threadIdx.x) & 7) + 776)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 194)] * weight_shared[((((int)threadIdx.x) & 7) + 784)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 386)] * weight_shared[((((int)threadIdx.x) & 7) + 784)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 195)] * weight_shared[((((int)threadIdx.x) & 7) + 792)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 387)] * weight_shared[((((int)threadIdx.x) & 7) + 792)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 200)] * weight_shared[((((int)threadIdx.x) & 7) + 832)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 392)] * weight_shared[((((int)threadIdx.x) & 7) + 832)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 201)] * weight_shared[((((int)threadIdx.x) & 7) + 840)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 393)] * weight_shared[((((int)threadIdx.x) & 7) + 840)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 202)] * weight_shared[((((int)threadIdx.x) & 7) + 848)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 394)] * weight_shared[((((int)threadIdx.x) & 7) + 848)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 203)] * weight_shared[((((int)threadIdx.x) & 7) + 856)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 395)] * weight_shared[((((int)threadIdx.x) & 7) + 856)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 4)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 196)] * weight_shared[((((int)threadIdx.x) & 7) + 32)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 5)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 197)] * weight_shared[((((int)threadIdx.x) & 7) + 40)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 6)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 198)] * weight_shared[((((int)threadIdx.x) & 7) + 48)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 7)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 199)] * weight_shared[((((int)threadIdx.x) & 7) + 56)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 12)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 204)] * weight_shared[((((int)threadIdx.x) & 7) + 96)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 13)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 205)] * weight_shared[((((int)threadIdx.x) & 7) + 104)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 14)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 206)] * weight_shared[((((int)threadIdx.x) & 7) + 112)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 15)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 207)] * weight_shared[((((int)threadIdx.x) & 7) + 120)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 100)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 292)] * weight_shared[((((int)threadIdx.x) & 7) + 416)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 101)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 293)] * weight_shared[((((int)threadIdx.x) & 7) + 424)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 102)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 294)] * weight_shared[((((int)threadIdx.x) & 7) + 432)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 103)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 295)] * weight_shared[((((int)threadIdx.x) & 7) + 440)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 108)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 300)] * weight_shared[((((int)threadIdx.x) & 7) + 480)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 109)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 301)] * weight_shared[((((int)threadIdx.x) & 7) + 488)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 110)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 302)] * weight_shared[((((int)threadIdx.x) & 7) + 496)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 111)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 303)] * weight_shared[((((int)threadIdx.x) & 7) + 504)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 196)] * weight_shared[((((int)threadIdx.x) & 7) + 800)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 388)] * weight_shared[((((int)threadIdx.x) & 7) + 800)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 197)] * weight_shared[((((int)threadIdx.x) & 7) + 808)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 389)] * weight_shared[((((int)threadIdx.x) & 7) + 808)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 198)] * weight_shared[((((int)threadIdx.x) & 7) + 816)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 390)] * weight_shared[((((int)threadIdx.x) & 7) + 816)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 199)] * weight_shared[((((int)threadIdx.x) & 7) + 824)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 391)] * weight_shared[((((int)threadIdx.x) & 7) + 824)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 204)] * weight_shared[((((int)threadIdx.x) & 7) + 864)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 396)] * weight_shared[((((int)threadIdx.x) & 7) + 864)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 205)] * weight_shared[((((int)threadIdx.x) & 7) + 872)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 397)] * weight_shared[((((int)threadIdx.x) & 7) + 872)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 206)] * weight_shared[((((int)threadIdx.x) & 7) + 880)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 398)] * weight_shared[((((int)threadIdx.x) & 7) + 880)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 207)] * weight_shared[((((int)threadIdx.x) & 7) + 888)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 399)] * weight_shared[((((int)threadIdx.x) & 7) + 888)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 32)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 224)] * weight_shared[((((int)threadIdx.x) & 7) + 128)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 33)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 225)] * weight_shared[((((int)threadIdx.x) & 7) + 136)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 34)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 226)] * weight_shared[((((int)threadIdx.x) & 7) + 144)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 35)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 227)] * weight_shared[((((int)threadIdx.x) & 7) + 152)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 40)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 232)] * weight_shared[((((int)threadIdx.x) & 7) + 192)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 41)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 233)] * weight_shared[((((int)threadIdx.x) & 7) + 200)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 42)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 234)] * weight_shared[((((int)threadIdx.x) & 7) + 208)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 43)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 235)] * weight_shared[((((int)threadIdx.x) & 7) + 216)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 128)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 320)] * weight_shared[((((int)threadIdx.x) & 7) + 512)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 129)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 321)] * weight_shared[((((int)threadIdx.x) & 7) + 520)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 130)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 322)] * weight_shared[((((int)threadIdx.x) & 7) + 528)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 131)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 323)] * weight_shared[((((int)threadIdx.x) & 7) + 536)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 136)] * weight_shared[((((int)threadIdx.x) & 7) + 576)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 328)] * weight_shared[((((int)threadIdx.x) & 7) + 576)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 137)] * weight_shared[((((int)threadIdx.x) & 7) + 584)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 329)] * weight_shared[((((int)threadIdx.x) & 7) + 584)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 138)] * weight_shared[((((int)threadIdx.x) & 7) + 592)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 330)] * weight_shared[((((int)threadIdx.x) & 7) + 592)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 139)] * weight_shared[((((int)threadIdx.x) & 7) + 600)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 331)] * weight_shared[((((int)threadIdx.x) & 7) + 600)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 224)] * weight_shared[((((int)threadIdx.x) & 7) + 896)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 416)] * weight_shared[((((int)threadIdx.x) & 7) + 896)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 225)] * weight_shared[((((int)threadIdx.x) & 7) + 904)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 417)] * weight_shared[((((int)threadIdx.x) & 7) + 904)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 226)] * weight_shared[((((int)threadIdx.x) & 7) + 912)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 418)] * weight_shared[((((int)threadIdx.x) & 7) + 912)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 227)] * weight_shared[((((int)threadIdx.x) & 7) + 920)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 419)] * weight_shared[((((int)threadIdx.x) & 7) + 920)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 232)] * weight_shared[((((int)threadIdx.x) & 7) + 960)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 424)] * weight_shared[((((int)threadIdx.x) & 7) + 960)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 233)] * weight_shared[((((int)threadIdx.x) & 7) + 968)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 425)] * weight_shared[((((int)threadIdx.x) & 7) + 968)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 234)] * weight_shared[((((int)threadIdx.x) & 7) + 976)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 426)] * weight_shared[((((int)threadIdx.x) & 7) + 976)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 235)] * weight_shared[((((int)threadIdx.x) & 7) + 984)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 427)] * weight_shared[((((int)threadIdx.x) & 7) + 984)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 36)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 228)] * weight_shared[((((int)threadIdx.x) & 7) + 160)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 37)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 229)] * weight_shared[((((int)threadIdx.x) & 7) + 168)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 38)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 230)] * weight_shared[((((int)threadIdx.x) & 7) + 176)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 39)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 231)] * weight_shared[((((int)threadIdx.x) & 7) + 184)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 44)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 236)] * weight_shared[((((int)threadIdx.x) & 7) + 224)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 45)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 237)] * weight_shared[((((int)threadIdx.x) & 7) + 232)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 46)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 238)] * weight_shared[((((int)threadIdx.x) & 7) + 240)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 47)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 239)] * weight_shared[((((int)threadIdx.x) & 7) + 248)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 132)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 324)] * weight_shared[((((int)threadIdx.x) & 7) + 544)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 133)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 325)] * weight_shared[((((int)threadIdx.x) & 7) + 552)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 134)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 326)] * weight_shared[((((int)threadIdx.x) & 7) + 560)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 135)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 327)] * weight_shared[((((int)threadIdx.x) & 7) + 568)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 140)] * weight_shared[((((int)threadIdx.x) & 7) + 608)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 332)] * weight_shared[((((int)threadIdx.x) & 7) + 608)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 141)] * weight_shared[((((int)threadIdx.x) & 7) + 616)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 333)] * weight_shared[((((int)threadIdx.x) & 7) + 616)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 142)] * weight_shared[((((int)threadIdx.x) & 7) + 624)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 334)] * weight_shared[((((int)threadIdx.x) & 7) + 624)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 143)] * weight_shared[((((int)threadIdx.x) & 7) + 632)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 335)] * weight_shared[((((int)threadIdx.x) & 7) + 632)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 228)] * weight_shared[((((int)threadIdx.x) & 7) + 928)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 420)] * weight_shared[((((int)threadIdx.x) & 7) + 928)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 229)] * weight_shared[((((int)threadIdx.x) & 7) + 936)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 421)] * weight_shared[((((int)threadIdx.x) & 7) + 936)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 230)] * weight_shared[((((int)threadIdx.x) & 7) + 944)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 422)] * weight_shared[((((int)threadIdx.x) & 7) + 944)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 231)] * weight_shared[((((int)threadIdx.x) & 7) + 952)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 423)] * weight_shared[((((int)threadIdx.x) & 7) + 952)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 236)] * weight_shared[((((int)threadIdx.x) & 7) + 992)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 428)] * weight_shared[((((int)threadIdx.x) & 7) + 992)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 237)] * weight_shared[((((int)threadIdx.x) & 7) + 1000)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 429)] * weight_shared[((((int)threadIdx.x) & 7) + 1000)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 238)] * weight_shared[((((int)threadIdx.x) & 7) + 1008)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 430)] * weight_shared[((((int)threadIdx.x) & 7) + 1008)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 239)] * weight_shared[((((int)threadIdx.x) & 7) + 1016)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 431)] * weight_shared[((((int)threadIdx.x) & 7) + 1016)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 64)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 256)] * weight_shared[((((int)threadIdx.x) & 7) + 256)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 65)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 257)] * weight_shared[((((int)threadIdx.x) & 7) + 264)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 66)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 258)] * weight_shared[((((int)threadIdx.x) & 7) + 272)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 67)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 259)] * weight_shared[((((int)threadIdx.x) & 7) + 280)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 72)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 264)] * weight_shared[((((int)threadIdx.x) & 7) + 320)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 73)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 265)] * weight_shared[((((int)threadIdx.x) & 7) + 328)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 74)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 266)] * weight_shared[((((int)threadIdx.x) & 7) + 336)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 75)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 267)] * weight_shared[((((int)threadIdx.x) & 7) + 344)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 160)] * weight_shared[((((int)threadIdx.x) & 7) + 640)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 352)] * weight_shared[((((int)threadIdx.x) & 7) + 640)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 161)] * weight_shared[((((int)threadIdx.x) & 7) + 648)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 353)] * weight_shared[((((int)threadIdx.x) & 7) + 648)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 162)] * weight_shared[((((int)threadIdx.x) & 7) + 656)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 354)] * weight_shared[((((int)threadIdx.x) & 7) + 656)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 163)] * weight_shared[((((int)threadIdx.x) & 7) + 664)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 355)] * weight_shared[((((int)threadIdx.x) & 7) + 664)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 168)] * weight_shared[((((int)threadIdx.x) & 7) + 704)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 360)] * weight_shared[((((int)threadIdx.x) & 7) + 704)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 169)] * weight_shared[((((int)threadIdx.x) & 7) + 712)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 361)] * weight_shared[((((int)threadIdx.x) & 7) + 712)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 170)] * weight_shared[((((int)threadIdx.x) & 7) + 720)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 362)] * weight_shared[((((int)threadIdx.x) & 7) + 720)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 171)] * weight_shared[((((int)threadIdx.x) & 7) + 728)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 363)] * weight_shared[((((int)threadIdx.x) & 7) + 728)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 256)] * weight_shared[((((int)threadIdx.x) & 7) + 1024)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 448)] * weight_shared[((((int)threadIdx.x) & 7) + 1024)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 257)] * weight_shared[((((int)threadIdx.x) & 7) + 1032)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 449)] * weight_shared[((((int)threadIdx.x) & 7) + 1032)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 258)] * weight_shared[((((int)threadIdx.x) & 7) + 1040)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 450)] * weight_shared[((((int)threadIdx.x) & 7) + 1040)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 259)] * weight_shared[((((int)threadIdx.x) & 7) + 1048)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 451)] * weight_shared[((((int)threadIdx.x) & 7) + 1048)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 264)] * weight_shared[((((int)threadIdx.x) & 7) + 1088)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 456)] * weight_shared[((((int)threadIdx.x) & 7) + 1088)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 265)] * weight_shared[((((int)threadIdx.x) & 7) + 1096)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 457)] * weight_shared[((((int)threadIdx.x) & 7) + 1096)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 266)] * weight_shared[((((int)threadIdx.x) & 7) + 1104)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 458)] * weight_shared[((((int)threadIdx.x) & 7) + 1104)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 267)] * weight_shared[((((int)threadIdx.x) & 7) + 1112)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 459)] * weight_shared[((((int)threadIdx.x) & 7) + 1112)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 68)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 260)] * weight_shared[((((int)threadIdx.x) & 7) + 288)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 69)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 261)] * weight_shared[((((int)threadIdx.x) & 7) + 296)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 70)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 262)] * weight_shared[((((int)threadIdx.x) & 7) + 304)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 71)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 263)] * weight_shared[((((int)threadIdx.x) & 7) + 312)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 76)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 268)] * weight_shared[((((int)threadIdx.x) & 7) + 352)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 77)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 269)] * weight_shared[((((int)threadIdx.x) & 7) + 360)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 78)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 270)] * weight_shared[((((int)threadIdx.x) & 7) + 368)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 79)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 271)] * weight_shared[((((int)threadIdx.x) & 7) + 376)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 164)] * weight_shared[((((int)threadIdx.x) & 7) + 672)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 356)] * weight_shared[((((int)threadIdx.x) & 7) + 672)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 165)] * weight_shared[((((int)threadIdx.x) & 7) + 680)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 357)] * weight_shared[((((int)threadIdx.x) & 7) + 680)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 166)] * weight_shared[((((int)threadIdx.x) & 7) + 688)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 358)] * weight_shared[((((int)threadIdx.x) & 7) + 688)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 167)] * weight_shared[((((int)threadIdx.x) & 7) + 696)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 359)] * weight_shared[((((int)threadIdx.x) & 7) + 696)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 172)] * weight_shared[((((int)threadIdx.x) & 7) + 736)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 364)] * weight_shared[((((int)threadIdx.x) & 7) + 736)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 173)] * weight_shared[((((int)threadIdx.x) & 7) + 744)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 365)] * weight_shared[((((int)threadIdx.x) & 7) + 744)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 174)] * weight_shared[((((int)threadIdx.x) & 7) + 752)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 366)] * weight_shared[((((int)threadIdx.x) & 7) + 752)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 175)] * weight_shared[((((int)threadIdx.x) & 7) + 760)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 367)] * weight_shared[((((int)threadIdx.x) & 7) + 760)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 260)] * weight_shared[((((int)threadIdx.x) & 7) + 1056)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 452)] * weight_shared[((((int)threadIdx.x) & 7) + 1056)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 261)] * weight_shared[((((int)threadIdx.x) & 7) + 1064)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 453)] * weight_shared[((((int)threadIdx.x) & 7) + 1064)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 262)] * weight_shared[((((int)threadIdx.x) & 7) + 1072)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 454)] * weight_shared[((((int)threadIdx.x) & 7) + 1072)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 263)] * weight_shared[((((int)threadIdx.x) & 7) + 1080)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 455)] * weight_shared[((((int)threadIdx.x) & 7) + 1080)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 268)] * weight_shared[((((int)threadIdx.x) & 7) + 1120)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 460)] * weight_shared[((((int)threadIdx.x) & 7) + 1120)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 269)] * weight_shared[((((int)threadIdx.x) & 7) + 1128)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 461)] * weight_shared[((((int)threadIdx.x) & 7) + 1128)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 270)] * weight_shared[((((int)threadIdx.x) & 7) + 1136)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 462)] * weight_shared[((((int)threadIdx.x) & 7) + 1136)]));
  conv2d_capsule_nhwijc_local[0] = (conv2d_capsule_nhwijc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 271)] * weight_shared[((((int)threadIdx.x) & 7) + 1144)]));
  conv2d_capsule_nhwijc_local[1] = (conv2d_capsule_nhwijc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 384) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + 463)] * weight_shared[((((int)threadIdx.x) & 7) + 1144)]));
  conv2d_capsule_nhwijc[(((((((((int)blockIdx.x) >> 7) * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((((int)blockIdx.x) & 127) >> 3) * 128)) + (((((int)threadIdx.x) & 15) >> 3) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_capsule_nhwijc_local[0];
  conv2d_capsule_nhwijc[((((((((((int)blockIdx.x) >> 7) * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((((int)blockIdx.x) & 127) >> 3) * 128)) + (((((int)threadIdx.x) & 15) >> 3) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + (((int)threadIdx.x) & 7)) + 2048)] = conv2d_capsule_nhwijc_local[1];
}

