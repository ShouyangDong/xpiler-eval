
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
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[14];
  __shared__ float PadInput_shared[4104];
  __shared__ float weight_shared[1152];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  conv2d_nhwc_local[8] = 0.000000e+00f;
  conv2d_nhwc_local[9] = 0.000000e+00f;
  conv2d_nhwc_local[10] = 0.000000e+00f;
  conv2d_nhwc_local[11] = 0.000000e+00f;
  conv2d_nhwc_local[12] = 0.000000e+00f;
  conv2d_nhwc_local[13] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 2; ++rc_0) {
    __syncthreads();
    float condval;
    if (((8 <= ((int)blockIdx.x)) && (8 <= ((int)threadIdx.x)))) {
      condval = inputs[(((((((((int)blockIdx.x) >> 3) * 28672) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[((int)threadIdx.x)] = condval;
    float condval_1;
    if ((8 <= ((int)blockIdx.x))) {
      condval_1 = inputs[(((((((((int)blockIdx.x) >> 3) * 28672) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2624)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 128)] = condval_1;
    float condval_2;
    if ((8 <= ((int)blockIdx.x))) {
      condval_2 = inputs[(((((((((int)blockIdx.x) >> 3) * 28672) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1600)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 256)] = condval_2;
    float condval_3;
    if (((1 <= (((((int)blockIdx.x) >> 3) * 8) + ((((int)threadIdx.x) + 384) / 456))) && (1 <= (((((int)threadIdx.x) >> 3) + 48) % 57)))) {
      condval_3 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 384) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 48) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 384)] = condval_3;
    PadInput_shared[(((int)threadIdx.x) + 512)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 512) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3200)];
    PadInput_shared[(((int)threadIdx.x) + 640)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 640) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2176)];
    PadInput_shared[(((int)threadIdx.x) + 768)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 768) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1152)];
    float condval_4;
    if ((1 <= (((((int)threadIdx.x) >> 3) + 55) % 57))) {
      condval_4 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 896) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 55) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 896)] = condval_4;
    PadInput_shared[(((int)threadIdx.x) + 1024)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1024) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2752)];
    PadInput_shared[(((int)threadIdx.x) + 1152)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1152) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1728)];
    float condval_5;
    if ((1 <= (((((int)threadIdx.x) >> 3) + 46) % 57))) {
      condval_5 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1280) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 46) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1280)] = condval_5;
    PadInput_shared[(((int)threadIdx.x) + 1408)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1408) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3328)];
    PadInput_shared[(((int)threadIdx.x) + 1536)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1536) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2304)];
    PadInput_shared[(((int)threadIdx.x) + 1664)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1664) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1280)];
    float condval_6;
    if ((1 <= (((((int)threadIdx.x) >> 3) + 53) % 57))) {
      condval_6 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1792) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 53) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1792)] = condval_6;
    PadInput_shared[(((int)threadIdx.x) + 1920)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 1920) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2880)];
    PadInput_shared[(((int)threadIdx.x) + 2048)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2048) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1856)];
    float condval_7;
    if ((1 <= (((((int)threadIdx.x) >> 3) + 44) % 57))) {
      condval_7 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2176) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 44) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2176)] = condval_7;
    PadInput_shared[(((int)threadIdx.x) + 2304)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2304) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3456)];
    PadInput_shared[(((int)threadIdx.x) + 2432)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2432) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2432)];
    PadInput_shared[(((int)threadIdx.x) + 2560)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2560) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1408)];
    float condval_8;
    if ((1 <= (((((int)threadIdx.x) >> 3) + 51) % 57))) {
      condval_8 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2688) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 51) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2688)] = condval_8;
    PadInput_shared[(((int)threadIdx.x) + 2816)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2816) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3008)];
    PadInput_shared[(((int)threadIdx.x) + 2944)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 2944) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1984)];
    float condval_9;
    if ((1 <= (((((int)threadIdx.x) >> 3) + 42) % 57))) {
      condval_9 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3072) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 42) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 3072)] = condval_9;
    PadInput_shared[(((int)threadIdx.x) + 3200)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3200) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3584)];
    PadInput_shared[(((int)threadIdx.x) + 3328)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3328) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2560)];
    PadInput_shared[(((int)threadIdx.x) + 3456)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3456) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1536)];
    float condval_10;
    if ((1 <= (((((int)threadIdx.x) >> 3) + 49) % 57))) {
      condval_10 = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3584) / 456) * 3584)) + ((((((int)threadIdx.x) >> 3) + 49) % 57) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 3584)] = condval_10;
    PadInput_shared[(((int)threadIdx.x) + 3712)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3712) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3136)];
    PadInput_shared[(((int)threadIdx.x) + 3840)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3840) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2112)];
    PadInput_shared[(((int)threadIdx.x) + 3968)] = inputs[((((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)threadIdx.x) + 3968) / 456) * 3584)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1088)];
    if (((int)threadIdx.x) < 8) {
      PadInput_shared[(((int)threadIdx.x) + 4096)] = inputs[((((((((int)blockIdx.x) >> 3) * 28672) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (rc_0 * 8)) + ((int)threadIdx.x)) + 28608)];
    }
    *(float2*)(weight_shared + (((int)threadIdx.x) * 2)) = *(float2*)(weight + ((((((((int)threadIdx.x) >> 6) * 2048) + (rc_0 * 1024)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(weight + (((((((((int)threadIdx.x) >> 6) * 2048) + (rc_0 * 1024)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 4096));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(weight + (((((((((int)threadIdx.x) >> 6) * 2048) + (rc_0 * 1024)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 8192));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(weight + (((((((((int)threadIdx.x) >> 6) * 2048) + (rc_0 * 1024)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 12288));
    if (((int)threadIdx.x) < 64) {
      *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(float2*)(weight + (((((rc_0 * 1024) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 16384));
    }
    __syncthreads();
    for (int rh_1 = 0; rh_1 < 3; ++rh_1) {
      for (int rw_2 = 0; rw_2 < 3; ++rw_2) {
        for (int rc_2 = 0; rc_2 < 8; ++rc_2) {
          conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2)] * weight_shared[((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7))]));
          conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2)] * weight_shared[(((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
          conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 64)] * weight_shared[((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7))]));
          conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 64)] * weight_shared[(((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
          conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 128)] * weight_shared[((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7))]));
          conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 128)] * weight_shared[(((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
          conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 192)] * weight_shared[((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7))]));
          conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 192)] * weight_shared[(((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
          conv2d_nhwc_local[8] = (conv2d_nhwc_local[8] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 256)] * weight_shared[((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7))]));
          conv2d_nhwc_local[9] = (conv2d_nhwc_local[9] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 256)] * weight_shared[(((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
          conv2d_nhwc_local[10] = (conv2d_nhwc_local[10] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 320)] * weight_shared[((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7))]));
          conv2d_nhwc_local[11] = (conv2d_nhwc_local[11] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 320)] * weight_shared[(((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
          conv2d_nhwc_local[12] = (conv2d_nhwc_local[12] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 384)] * weight_shared[((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7))]));
          conv2d_nhwc_local[13] = (conv2d_nhwc_local[13] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 912) + (rh_1 * 456)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + (rw_2 * 8)) + rc_2) + 384)] * weight_shared[(((((rh_1 * 384) + (rw_2 * 128)) + (rc_2 * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
        }
      }
    }
  }
  conv2d_nhwc[((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 512)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 520)] = conv2d_nhwc_local[3];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 1024)] = conv2d_nhwc_local[4];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 1032)] = conv2d_nhwc_local[5];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 1536)] = conv2d_nhwc_local[6];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 1544)] = conv2d_nhwc_local[7];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 2048)] = conv2d_nhwc_local[8];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 2056)] = conv2d_nhwc_local[9];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 2560)] = conv2d_nhwc_local[10];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 2568)] = conv2d_nhwc_local[11];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 3072)] = conv2d_nhwc_local[12];
  conv2d_nhwc[(((((((((int)blockIdx.x) >> 3) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 3080)] = conv2d_nhwc_local[13];
}

