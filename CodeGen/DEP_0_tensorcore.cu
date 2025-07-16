
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
  __shared__ float PadInput_shared[3456];
  __shared__ float placeholder_shared[288];
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
  float condval;
  if (((7 <= ((int)blockIdx.x)) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5))))) {
    condval = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3616)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  float condval_1;
  if ((7 <= ((int)blockIdx.x))) {
    condval_1 = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3488)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 128)] = condval_1;
  float condval_2;
  if ((7 <= ((int)blockIdx.x))) {
    condval_2 = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3360)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 256)] = condval_2;
  float condval_3;
  if ((7 <= ((int)blockIdx.x))) {
    condval_3 = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3232)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 384)] = condval_3;
  float condval_4;
  if ((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 512) / 576))) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)) < 113))) {
    condval_4 = placeholder[(((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 512) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((((((int)threadIdx.x) >> 5) + 16) % 18) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 512)] = condval_4;
  PadInput_shared[(((int)threadIdx.x) + 640)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 640) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3552)];
  PadInput_shared[(((int)threadIdx.x) + 768)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 768) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3424)];
  PadInput_shared[(((int)threadIdx.x) + 896)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 896) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3296)];
  float condval_5;
  if (((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)) < 99)) {
    condval_5 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1024) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3168)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1024)] = condval_5;
  float condval_6;
  if ((1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)))) {
    condval_6 = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) + 3552)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1152)] = condval_6;
  PadInput_shared[(((int)threadIdx.x) + 1280)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1280) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3488)];
  PadInput_shared[(((int)threadIdx.x) + 1408)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1408) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3360)];
  PadInput_shared[(((int)threadIdx.x) + 1536)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1536) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3232)];
  float condval_7;
  if (((1 <= (((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18))) && ((((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)) < 113))) {
    condval_7 = placeholder[(((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1664) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((((((int)threadIdx.x) >> 5) + 16) % 18) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1664)] = condval_7;
  PadInput_shared[(((int)threadIdx.x) + 1792)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1792) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3552)];
  PadInput_shared[(((int)threadIdx.x) + 1920)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1920) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3424)];
  PadInput_shared[(((int)threadIdx.x) + 2048)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2048) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3296)];
  float condval_8;
  if (((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)) < 99)) {
    condval_8 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2176) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3168)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2176)] = condval_8;
  float condval_9;
  if ((1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)))) {
    condval_9 = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) + 10720)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2304)] = condval_9;
  PadInput_shared[(((int)threadIdx.x) + 2432)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2432) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3488)];
  PadInput_shared[(((int)threadIdx.x) + 2560)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2560) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3360)];
  PadInput_shared[(((int)threadIdx.x) + 2688)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2688) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3232)];
  float condval_10;
  if (((((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 2816) / 576)) < 113) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)) < 113))) {
    condval_10 = placeholder[(((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2816) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((((((int)threadIdx.x) >> 5) + 16) % 18) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2816)] = condval_10;
  float condval_11;
  if (((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 2944) / 576)) < 113)) {
    condval_11 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2944) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3552)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2944)] = condval_11;
  float condval_12;
  if (((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 3072) / 576)) < 113)) {
    condval_12 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 3072) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3072)] = condval_12;
  float condval_13;
  if (((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 3200) / 576)) < 113)) {
    condval_13 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 3200) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3296)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3200)] = condval_13;
  float condval_14;
  if ((((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 3328) / 576)) < 113) && ((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)) < 99))) {
    condval_14 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 3328) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3168)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3328)] = condval_14;
  *(float2*)(placeholder_shared + (((int)threadIdx.x) * 2)) = *(float2*)(placeholder_1 + (((int)threadIdx.x) * 2));
  if (((int)threadIdx.x) < 16) {
    *(float2*)(placeholder_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(placeholder_1 + ((((int)threadIdx.x) * 2) + 256));
  }
  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2))] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 32)] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 33)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 64)] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 65)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 96)] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 97)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 129)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 160)] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 161)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 192)] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 193)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 224)] * placeholder_shared[((((int)threadIdx.x) & 15) * 2)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 225)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 32)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 33)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 64)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 65)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 96)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 97)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 128)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 129)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 160)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 161)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 192)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 193)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 224)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 225)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 256)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 257)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 64)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 65)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 96)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 97)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 128)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 129)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 160)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 161)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 192)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 193)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 224)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 225)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 256)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 257)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 288)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 289)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 576)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 577)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 608)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 609)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 640)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 641)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 672)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 673)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 704)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 705)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 736)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 737)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 768)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 769)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 800)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 801)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 608)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 609)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 640)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 641)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 672)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 673)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 704)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 705)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 736)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 737)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 768)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 769)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 800)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 801)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 832)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 833)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 640)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 641)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 672)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 673)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 704)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 705)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 736)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 737)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 768)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 769)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 800)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 801)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 832)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 833)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 864)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 865)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1152)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1153)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1184)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1185)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1216)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1217)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1248)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1249)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1280)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1281)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1312)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1313)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1344)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1345)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1376)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1377)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1184)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1185)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1216)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1217)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1248)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1249)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1280)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1281)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1312)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1313)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1344)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1345)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1376)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1377)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1408)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1409)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1216)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1217)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1248)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1249)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1280)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1281)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1312)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1313)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1344)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1345)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1376)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1377)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1408)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1409)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1440)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 576) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1441)] * placeholder_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 1)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 32)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 33)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 64)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 65)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 96)] = depth_conv2d_nhwc_local[6];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 97)] = depth_conv2d_nhwc_local[7];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 128)] = depth_conv2d_nhwc_local[8];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 129)] = depth_conv2d_nhwc_local[9];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 160)] = depth_conv2d_nhwc_local[10];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 161)] = depth_conv2d_nhwc_local[11];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 192)] = depth_conv2d_nhwc_local[12];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 193)] = depth_conv2d_nhwc_local[13];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 224)] = depth_conv2d_nhwc_local[14];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 15) * 2)) + 225)] = depth_conv2d_nhwc_local[15];
}

