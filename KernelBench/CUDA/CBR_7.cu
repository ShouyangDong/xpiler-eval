
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
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float Conv2dOutput_local[7];
  __shared__ float PaddedInput_shared[3456];
  __shared__ float kernel_shared[4608];
  Conv2dOutput_local[0] = 0.000000e+00f;
  Conv2dOutput_local[1] = 0.000000e+00f;
  Conv2dOutput_local[2] = 0.000000e+00f;
  Conv2dOutput_local[3] = 0.000000e+00f;
  Conv2dOutput_local[4] = 0.000000e+00f;
  Conv2dOutput_local[5] = 0.000000e+00f;
  Conv2dOutput_local[6] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 16; ++rc_0) {
    __syncthreads();
    PaddedInput_shared[((int)threadIdx.x)] = 0.000000e+00f;
    float condval;
    if (((((((int)threadIdx.x) >> 5) == 1) && (1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 64) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 64)] = condval;
    float condval_1;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_1 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 128) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 128)] = condval_1;
    float condval_2;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_2 = data[(((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 3072)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 192)] = condval_2;
    float condval_3;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_3 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 256) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 256)] = condval_3;
    float condval_4;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_4 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 320) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 320)] = condval_4;
    float condval_5;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_5 = data[(((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 10240)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 384)] = condval_5;
    float condval_6;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_6 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 448) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 448)] = condval_6;
    float condval_7;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_7 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 512) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 512)] = condval_7;
    float condval_8;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_8 = data[(((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 17408)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 576)] = condval_8;
    float condval_9;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_9 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 640) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 640)] = condval_9;
    float condval_10;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_10 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 704) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 704)] = condval_10;
    PaddedInput_shared[(((int)threadIdx.x) + 768)] = 0.000000e+00f;
    float condval_11;
    if (((((3 <= (((((int)threadIdx.x) >> 5) + 26) % 27)) && (((((int)threadIdx.x) + 832) % 864) < 768)) && (1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_11 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 832) / 864) * 25088)) + (((((((int)threadIdx.x) >> 5) + 26) % 27) / 3) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_11 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 832)] = condval_11;
    PaddedInput_shared[(((int)threadIdx.x) + 896)] = 0.000000e+00f;
    float condval_12;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_12 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 960) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 512)];
    } else {
      condval_12 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 960)] = condval_12;
    float condval_13;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_13 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1024) / 864) * 25088)) + (((((int)threadIdx.x) + 160) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_13 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1024)] = condval_13;
    float condval_14;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_14 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1088) / 864) * 25088)) + (((((int)threadIdx.x) + 224) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_14 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1088)] = condval_14;
    float condval_15;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_15 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1152) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 6656)];
    } else {
      condval_15 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1152)] = condval_15;
    float condval_16;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_16 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1216) / 864) * 25088)) + (((((int)threadIdx.x) + 352) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_16 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1216)] = condval_16;
    float condval_17;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_17 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1280) / 864) * 25088)) + (((((int)threadIdx.x) + 416) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_17 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1280)] = condval_17;
    float condval_18;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_18 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1344) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 13824)];
    } else {
      condval_18 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = condval_18;
    float condval_19;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_19 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1408) / 864) * 25088)) + (((((int)threadIdx.x) + 544) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_19 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1408)] = condval_19;
    float condval_20;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_20 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1472) / 864) * 25088)) + (((((int)threadIdx.x) + 608) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_20 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1472)] = condval_20;
    float condval_21;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_21 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1536) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 20992)];
    } else {
      condval_21 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1536)] = condval_21;
    float condval_22;
    if ((((((int)threadIdx.x) < 32) && (1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_22 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1600) / 864) * 25088)) + (((((int)threadIdx.x) + 736) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + ((int)threadIdx.x)) - 3072)];
    } else {
      condval_22 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1600)] = condval_22;
    PaddedInput_shared[(((int)threadIdx.x) + 1664)] = 0.000000e+00f;
    PaddedInput_shared[(((int)threadIdx.x) + 1728)] = 0.000000e+00f;
    float condval_23;
    if (((((((int)threadIdx.x) >> 5) == 1) && (1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_23 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1792) / 864) * 25088)) + (((((int)threadIdx.x) + 64) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_23 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1792)] = condval_23;
    float condval_24;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_24 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1856) / 864) * 25088)) + (((((int)threadIdx.x) + 128) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_24 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1856)] = condval_24;
    float condval_25;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_25 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1920) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 3072)];
    } else {
      condval_25 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1920)] = condval_25;
    float condval_26;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_26 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 1984) / 864) * 25088)) + (((((int)threadIdx.x) + 256) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_26 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 1984)] = condval_26;
    float condval_27;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_27 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2048) / 864) * 25088)) + (((((int)threadIdx.x) + 320) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_27 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2048)] = condval_27;
    float condval_28;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_28 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2112) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 10240)];
    } else {
      condval_28 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2112)] = condval_28;
    float condval_29;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_29 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2176) / 864) * 25088)) + (((((int)threadIdx.x) + 448) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_29 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2176)] = condval_29;
    float condval_30;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_30 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2240) / 864) * 25088)) + (((((int)threadIdx.x) + 512) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_30 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2240)] = condval_30;
    float condval_31;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_31 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2304) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 17408)];
    } else {
      condval_31 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2304)] = condval_31;
    float condval_32;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_32 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2368) / 864) * 25088)) + (((((int)threadIdx.x) + 640) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_32 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2368)] = condval_32;
    float condval_33;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_33 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2432) / 864) * 25088)) + (((((int)threadIdx.x) + 704) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_33 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2432)] = condval_33;
    PaddedInput_shared[(((int)threadIdx.x) + 2496)] = 0.000000e+00f;
    float condval_34;
    if (((((3 <= (((((int)threadIdx.x) >> 5) + 26) % 27)) && (((((int)threadIdx.x) + 832) % 864) < 768)) && (1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_34 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2560) / 864) * 25088)) + (((((((int)threadIdx.x) >> 5) + 26) % 27) / 3) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_34 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2560)] = condval_34;
    PaddedInput_shared[(((int)threadIdx.x) + 2624)] = 0.000000e+00f;
    float condval_35;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_35 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2688) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 512)];
    } else {
      condval_35 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2688)] = condval_35;
    float condval_36;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_36 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2752) / 864) * 25088)) + (((((int)threadIdx.x) + 160) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_36 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2752)] = condval_36;
    float condval_37;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_37 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2816) / 864) * 25088)) + (((((int)threadIdx.x) + 224) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_37 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2816)] = condval_37;
    float condval_38;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_38 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2880) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 6656)];
    } else {
      condval_38 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2880)] = condval_38;
    float condval_39;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_39 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 2944) / 864) * 25088)) + (((((int)threadIdx.x) + 352) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_39 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 2944)] = condval_39;
    float condval_40;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_40 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 3008) / 864) * 25088)) + (((((int)threadIdx.x) + 416) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_40 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 3008)] = condval_40;
    float condval_41;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_41 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 3072) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 13824)];
    } else {
      condval_41 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 3072)] = condval_41;
    float condval_42;
    if (((1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_42 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 3136) / 864) * 25088)) + (((((int)threadIdx.x) + 544) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + ((((((int)threadIdx.x) >> 5) + 2) % 3) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
    } else {
      condval_42 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 3136)] = condval_42;
    float condval_43;
    if ((((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)) < 7)) {
      condval_43 = data[(((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 3200) / 864) * 25088)) + (((((int)threadIdx.x) + 608) / 96) * 3584)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
    } else {
      condval_43 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 3200)] = condval_43;
    float condval_44;
    if ((1 <= ((((int)threadIdx.x) >> 5) + ((((int)blockIdx.x) % 224) >> 5)))) {
      condval_44 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 3264) / 864) * 25088)) + ((((int)threadIdx.x) >> 5) * 512)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 20992)];
    } else {
      condval_44 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 3264)] = condval_44;
    float condval_45;
    if ((((((int)threadIdx.x) < 32) && (1 <= (((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && ((((((int)blockIdx.x) % 224) >> 5) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 8))) {
      condval_45 = data[((((((((((int)blockIdx.x) / 224) * 100352) + (((((int)threadIdx.x) + 3328) / 864) * 25088)) + (((((int)threadIdx.x) + 736) / 96) * 3584)) + (((((int)blockIdx.x) % 224) >> 5) * 512)) + (rc_0 * 32)) + ((int)threadIdx.x)) - 3072)];
    } else {
      condval_45 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 3328)] = condval_45;
    PaddedInput_shared[(((int)threadIdx.x) + 3392)] = 0.000000e+00f;
    *(float2*)(kernel_shared + (((int)threadIdx.x) * 2)) = *(float2*)(kernel + ((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 4096));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 8192));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 384)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 12288));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 262144));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 640)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 266240));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 270336));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 896)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 274432));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 524288));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1152)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 528384));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1280)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 532480));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1408)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 536576));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1536)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 786432));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1664)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 790528));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1792)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 794624));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1920)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 798720));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2048)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1048576));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2176)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1052672));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2304)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1056768));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2432)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1060864));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2560)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1310720));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2688)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1314816));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2816)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1318912));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 2944)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1323008));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3072)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1572864));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3200)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1576960));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3328)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1581056));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3456)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1585152));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3584)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1835008));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3712)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1839104));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3840)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1843200));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 3968)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1847296));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 4096)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2097152));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 4224)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2101248));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 4352)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2105344));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 4480)) = *(float2*)(kernel + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2109440));
    __syncthreads();
    for (int rx_1 = 0; rx_1 < 3; ++rx_1) {
      for (int rc_1 = 0; rc_1 < 4; ++rc_1) {
        for (int yy_3 = 0; yy_3 < 7; ++yy_3) {
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[(((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8))] * kernel_shared[(((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15))]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 1)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 16)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 2)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 32)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 3)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 48)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 4)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 64)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 5)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 80)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 6)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 96)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 7)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 112)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 96)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1536)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 97)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1552)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 98)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1568)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 99)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1584)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 100)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1600)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 101)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1616)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 102)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1632)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 103)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 1648)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 192)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3072)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 193)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3088)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 194)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3104)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 195)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3120)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 196)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3136)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 197)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3152)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 198)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3168)]));
          Conv2dOutput_local[yy_3] = (Conv2dOutput_local[yy_3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 864) + (yy_3 * 96)) + (rx_1 * 32)) + (rc_1 * 8)) + 199)] * kernel_shared[((((rx_1 * 512) + (rc_1 * 128)) + (((int)threadIdx.x) & 15)) + 3184)]));
        }
      }
    }
  }
  compute[(((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 4) * 25088)) + ((((int)blockIdx.x) % 224) * 16)) + (((int)threadIdx.x) & 15))] = max((((Conv2dOutput_local[0] + bias[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 4) * 25088)) + ((((int)blockIdx.x) % 224) * 16)) + (((int)threadIdx.x) & 15)) + 3584)] = max((((Conv2dOutput_local[1] + bias[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 4) * 25088)) + ((((int)blockIdx.x) % 224) * 16)) + (((int)threadIdx.x) & 15)) + 7168)] = max((((Conv2dOutput_local[2] + bias[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 4) * 25088)) + ((((int)blockIdx.x) % 224) * 16)) + (((int)threadIdx.x) & 15)) + 10752)] = max((((Conv2dOutput_local[3] + bias[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 4) * 25088)) + ((((int)blockIdx.x) % 224) * 16)) + (((int)threadIdx.x) & 15)) + 14336)] = max((((Conv2dOutput_local[4] + bias[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 4) * 25088)) + ((((int)blockIdx.x) % 224) * 16)) + (((int)threadIdx.x) & 15)) + 17920)] = max((((Conv2dOutput_local[5] + bias[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.x) / 224) * 100352) + ((((int)threadIdx.x) >> 4) * 25088)) + ((((int)blockIdx.x) % 224) * 16)) + (((int)threadIdx.x) & 15)) + 21504)] = max((((Conv2dOutput_local[6] + bias[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((((int)blockIdx.x) & 31) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
}

