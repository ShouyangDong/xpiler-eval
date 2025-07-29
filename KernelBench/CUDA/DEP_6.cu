
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
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[7];
  __shared__ float PadInput_shared[2880];
  __shared__ float placeholder_shared[288];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  float condval;
  if (((16 <= (((int)blockIdx.x) % 112)) && (32 <= ((int)threadIdx.x)))) {
    condval = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 7680)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  float condval_1;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_1 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 6656)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 64)] = condval_1;
  float condval_2;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_2 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 5632)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 128)] = condval_2;
  float condval_3;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_3 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 4608)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 192)] = condval_3;
  float condval_4;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_4 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 3584)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 256)] = condval_4;
  float condval_5;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_5 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 2560)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 320)] = condval_5;
  float condval_6;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_6 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 1536)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 384)] = condval_6;
  float condval_7;
  if (((16 <= (((int)blockIdx.x) % 112)) && (1 <= (((((int)threadIdx.x) >> 5) + 14) % 15)))) {
    condval_7 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 448) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((((int)threadIdx.x) >> 5) + 14) % 15) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 7680)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 448)] = condval_7;
  float condval_8;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_8 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 512) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 7168)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 512)] = condval_8;
  float condval_9;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_9 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 576) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 6144)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 576)] = condval_9;
  float condval_10;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_10 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 640) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 5120)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 640)] = condval_10;
  float condval_11;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_11 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 704) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 4096)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 704)] = condval_11;
  float condval_12;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_12 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 768) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 3072)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 768)] = condval_12;
  float condval_13;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_13 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 832) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 2048)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 832)] = condval_13;
  float condval_14;
  if ((16 <= (((int)blockIdx.x) % 112))) {
    condval_14 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 896) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 1024)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 896)] = condval_14;
  if (((int)threadIdx.x) < 48) {
    *(float2*)(placeholder_shared + (((int)threadIdx.x) * 2)) = *(float2*)(placeholder_1 + ((((((int)threadIdx.x) >> 4) * 512) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 15) * 2)));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float condval_15;
  if ((32 <= ((int)threadIdx.x))) {
    condval_15 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 512)];
  } else {
    condval_15 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 960)] = condval_15;
  PadInput_shared[(((int)threadIdx.x) + 1024)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 512)];
  PadInput_shared[(((int)threadIdx.x) + 1088)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 1536)];
  PadInput_shared[(((int)threadIdx.x) + 1152)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 2560)];
  PadInput_shared[(((int)threadIdx.x) + 1216)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
  PadInput_shared[(((int)threadIdx.x) + 1280)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 4608)];
  PadInput_shared[(((int)threadIdx.x) + 1344)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 5632)];
  float condval_16;
  if ((1 <= (((((int)threadIdx.x) >> 5) + 14) % 15))) {
    condval_16 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 448) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((((int)threadIdx.x) >> 5) + 14) % 15) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) - 512)];
  } else {
    condval_16 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1408)] = condval_16;
  PadInput_shared[(((int)threadIdx.x) + 1472)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 512) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31))];
  PadInput_shared[(((int)threadIdx.x) + 1536)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 576) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 1024)];
  PadInput_shared[(((int)threadIdx.x) + 1600)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 640) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
  PadInput_shared[(((int)threadIdx.x) + 1664)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 704) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 3072)];
  PadInput_shared[(((int)threadIdx.x) + 1728)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 768) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
  PadInput_shared[(((int)threadIdx.x) + 1792)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 832) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 5120)];
  PadInput_shared[(((int)threadIdx.x) + 1856)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 896) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
  if (((int)threadIdx.x) < 48) {
    *(float2*)(placeholder_shared + ((((int)threadIdx.x) * 2) + 96)) = *(float2*)(placeholder_1 + (((((((int)threadIdx.x) >> 4) * 512) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1536));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float condval_17;
  if ((32 <= ((int)threadIdx.x))) {
    condval_17 = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 6656)];
  } else {
    condval_17 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1920)] = condval_17;
  PadInput_shared[(((int)threadIdx.x) + 1984)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 7680)];
  PadInput_shared[(((int)threadIdx.x) + 2048)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 8704)];
  PadInput_shared[(((int)threadIdx.x) + 2112)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 9728)];
  PadInput_shared[(((int)threadIdx.x) + 2176)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 10752)];
  PadInput_shared[(((int)threadIdx.x) + 2240)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 11776)];
  PadInput_shared[(((int)threadIdx.x) + 2304)] = placeholder[(((((((((int)blockIdx.x) / 112) * 200704) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 12800)];
  float condval_18;
  if ((1 <= (((((int)threadIdx.x) >> 5) + 14) % 15))) {
    condval_18 = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 448) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((((int)threadIdx.x) >> 5) + 14) % 15) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 6656)];
  } else {
    condval_18 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2368)] = condval_18;
  PadInput_shared[(((int)threadIdx.x) + 2432)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 512) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
  PadInput_shared[(((int)threadIdx.x) + 2496)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 576) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
  PadInput_shared[(((int)threadIdx.x) + 2560)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 640) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 9216)];
  PadInput_shared[(((int)threadIdx.x) + 2624)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 704) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 10240)];
  PadInput_shared[(((int)threadIdx.x) + 2688)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 768) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 11264)];
  PadInput_shared[(((int)threadIdx.x) + 2752)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 832) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
  PadInput_shared[(((int)threadIdx.x) + 2816)] = placeholder[((((((((((int)blockIdx.x) / 112) * 200704) + (((((int)threadIdx.x) + 896) / 480) * 100352)) + (((((int)blockIdx.x) % 112) >> 4) * 14336)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 13312)];
  if (((int)threadIdx.x) < 48) {
    *(float2*)(placeholder_shared + ((((int)threadIdx.x) * 2) + 192)) = *(float2*)(placeholder_1 + (((((((int)threadIdx.x) >> 4) * 512) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3072));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31))] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 64)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 128)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 192)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 256)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 320)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[(((int)threadIdx.x) & 31)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 32)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 96)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 160)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 224)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 288)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 352)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 31) + 32)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 64)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 192)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 256)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 320)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 448)] * placeholder_shared[((((int)threadIdx.x) & 31) + 64)]));
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 960)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1152)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1216)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 96)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 992)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1056)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1184)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1248)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1312)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1376)] * placeholder_shared[((((int)threadIdx.x) & 31) + 128)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1024)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1088)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1152)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1216)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1280)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1408)] * placeholder_shared[((((int)threadIdx.x) & 31) + 160)]));
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1920)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2112)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2176)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 192)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1952)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2080)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2144)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2208)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2272)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2336)] * placeholder_shared[((((int)threadIdx.x) & 31) + 224)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 1984)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2048)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2112)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2176)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2304)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 480) + (((int)threadIdx.x) & 31)) + 2368)] * placeholder_shared[((((int)threadIdx.x) & 31) + 256)]));
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 112) * 50176) + ((((int)threadIdx.x) >> 5) * 25088)) + (((((int)blockIdx.x) % 112) >> 4) * 3584)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 112) * 50176) + ((((int)threadIdx.x) >> 5) * 25088)) + (((((int)blockIdx.x) % 112) >> 4) * 3584)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 512)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 112) * 50176) + ((((int)threadIdx.x) >> 5) * 25088)) + (((((int)blockIdx.x) % 112) >> 4) * 3584)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 1024)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 112) * 50176) + ((((int)threadIdx.x) >> 5) * 25088)) + (((((int)blockIdx.x) % 112) >> 4) * 3584)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 1536)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 112) * 50176) + ((((int)threadIdx.x) >> 5) * 25088)) + (((((int)blockIdx.x) % 112) >> 4) * 3584)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 2048)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 112) * 50176) + ((((int)threadIdx.x) >> 5) * 25088)) + (((((int)blockIdx.x) % 112) >> 4) * 3584)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 2560)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[(((((((((int)blockIdx.x) / 112) * 50176) + ((((int)threadIdx.x) >> 5) * 25088)) + (((((int)blockIdx.x) % 112) >> 4) * 3584)) + ((((int)blockIdx.x) & 15) * 32)) + (((int)threadIdx.x) & 31)) + 3072)] = depth_conv2d_nhwc_local[6];
}

