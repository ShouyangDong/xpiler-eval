
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
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_transpose_nhwc_local[4];
  __shared__ float PadInput_shared[960];
  __shared__ float weight_shared[8192];
  conv2d_transpose_nhwc_local[0] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[1] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[2] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[3] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 8; ++rc_0) {
    __syncthreads();
    float condval;
    if ((((1 <= (((((int)threadIdx.x) * 3) / 320) + (((int)blockIdx.x) >> 3))) && (32 <= ((((int)threadIdx.x) * 3) % 320))) && (((((int)threadIdx.x) * 3) % 320) < 288))) {
      condval = inputs[((((((((((int)threadIdx.x) * 3) / 320) * 2048) + ((((int)blockIdx.x) >> 3) * 2048)) + ((((((int)threadIdx.x) * 3) % 320) >> 5) * 256)) + (rc_0 * 32)) + ((((int)threadIdx.x) * 3) & 31)) - 2304)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) * 3)] = condval;
    float condval_1;
    if ((((1 <= ((((((int)threadIdx.x) * 3) + 1) / 320) + (((int)blockIdx.x) >> 3))) && (32 <= (((((int)threadIdx.x) * 3) + 1) % 320))) && ((((((int)threadIdx.x) * 3) + 1) % 320) < 288))) {
      condval_1 = inputs[(((((((((((int)threadIdx.x) * 3) + 1) / 320) * 2048) + ((((int)blockIdx.x) >> 3) * 2048)) + (((((((int)threadIdx.x) * 3) + 1) % 320) >> 5) * 256)) + (rc_0 * 32)) + (((((int)threadIdx.x) * 3) + 1) & 31)) - 2304)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 1)] = condval_1;
    float condval_2;
    if ((((1 <= ((((((int)threadIdx.x) * 3) + 2) / 320) + (((int)blockIdx.x) >> 3))) && (32 <= (((((int)threadIdx.x) * 3) + 2) % 320))) && ((((((int)threadIdx.x) * 3) + 2) % 320) < 288))) {
      condval_2 = inputs[(((((((((((int)threadIdx.x) * 3) + 2) / 320) * 2048) + ((((int)blockIdx.x) >> 3) * 2048)) + (((((((int)threadIdx.x) * 3) + 2) % 320) >> 5) * 256)) + (rc_0 * 32)) + (((((int)threadIdx.x) * 3) + 2) & 31)) - 2304)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 2)] = condval_2;
    float condval_3;
    if ((((((((((int)threadIdx.x) * 3) + 384) / 320) + (((int)blockIdx.x) >> 3)) < 9) && (1 <= ((((((int)threadIdx.x) * 3) >> 5) + 2) % 10))) && ((((((int)threadIdx.x) * 3) + 64) % 320) < 288))) {
      condval_3 = inputs[(((((((((((int)threadIdx.x) * 3) + 384) / 320) * 2048) + ((((int)blockIdx.x) >> 3) * 2048)) + (((((((int)threadIdx.x) * 3) >> 5) + 2) % 10) * 256)) + (rc_0 * 32)) + ((((int)threadIdx.x) * 3) & 31)) - 2304)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 384)] = condval_3;
    float condval_4;
    if ((((((((((int)threadIdx.x) * 3) + 385) / 320) + (((int)blockIdx.x) >> 3)) < 9) && (1 <= (((((((int)threadIdx.x) * 3) + 1) >> 5) + 2) % 10))) && ((((((int)threadIdx.x) * 3) + 65) % 320) < 288))) {
      condval_4 = inputs[(((((((((((int)threadIdx.x) * 3) + 385) / 320) * 2048) + ((((int)blockIdx.x) >> 3) * 2048)) + ((((((((int)threadIdx.x) * 3) + 1) >> 5) + 2) % 10) * 256)) + (rc_0 * 32)) + (((((int)threadIdx.x) * 3) + 1) & 31)) - 2304)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 385)] = condval_4;
    float condval_5;
    if ((((((((((int)threadIdx.x) * 3) + 386) / 320) + (((int)blockIdx.x) >> 3)) < 9) && (1 <= (((((((int)threadIdx.x) * 3) + 2) >> 5) + 2) % 10))) && ((((((int)threadIdx.x) * 3) + 66) % 320) < 288))) {
      condval_5 = inputs[(((((((((((int)threadIdx.x) * 3) + 386) / 320) * 2048) + ((((int)blockIdx.x) >> 3) * 2048)) + ((((((((int)threadIdx.x) * 3) + 2) >> 5) + 2) % 10) * 256)) + (rc_0 * 32)) + (((((int)threadIdx.x) * 3) + 2) & 31)) - 2304)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 3) + 386)] = condval_5;
    if (((int)threadIdx.x) < 64) {
      float condval_6;
      if (((((int)blockIdx.x) < 56) && (((int)threadIdx.x) < 54))) {
        condval_6 = inputs[((((((((int)blockIdx.x) >> 3) * 2048) + (((((int)threadIdx.x) * 3) >> 5) * 256)) + (rc_0 * 32)) + ((((int)threadIdx.x) * 3) & 31)) + 2816)];
      } else {
        condval_6 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 3) + 768)] = condval_6;
      float condval_7;
      if (((((int)blockIdx.x) < 56) && (((int)threadIdx.x) < 53))) {
        condval_7 = inputs[((((((((int)blockIdx.x) >> 3) * 2048) + ((((((int)threadIdx.x) * 3) + 1) >> 5) * 256)) + (rc_0 * 32)) + (((((int)threadIdx.x) * 3) + 1) & 31)) + 2816)];
      } else {
        condval_7 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 3) + 769)] = condval_7;
      float condval_8;
      if (((((int)blockIdx.x) < 56) && (((int)threadIdx.x) < 53))) {
        condval_8 = inputs[((((((((int)blockIdx.x) >> 3) * 2048) + ((((((int)threadIdx.x) * 3) + 2) >> 5) * 256)) + (rc_0 * 32)) + (((((int)threadIdx.x) * 3) + 2) & 31)) + 2816)];
      } else {
        condval_8 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 3) + 770)] = condval_8;
    }
    weight_shared[((int)threadIdx.x)] = weight[((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    weight_shared[(((int)threadIdx.x) + 128)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 1024)];
    weight_shared[(((int)threadIdx.x) + 256)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 2048)];
    weight_shared[(((int)threadIdx.x) + 384)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 3072)];
    weight_shared[(((int)threadIdx.x) + 512)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 32768)];
    weight_shared[(((int)threadIdx.x) + 640)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 33792)];
    weight_shared[(((int)threadIdx.x) + 768)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 34816)];
    weight_shared[(((int)threadIdx.x) + 896)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 35840)];
    weight_shared[(((int)threadIdx.x) + 1024)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 65536)];
    weight_shared[(((int)threadIdx.x) + 1152)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 66560)];
    weight_shared[(((int)threadIdx.x) + 1280)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 67584)];
    weight_shared[(((int)threadIdx.x) + 1408)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 68608)];
    weight_shared[(((int)threadIdx.x) + 1536)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 98304)];
    weight_shared[(((int)threadIdx.x) + 1664)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 99328)];
    weight_shared[(((int)threadIdx.x) + 1792)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 100352)];
    weight_shared[(((int)threadIdx.x) + 1920)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 101376)];
    weight_shared[(((int)threadIdx.x) + 2048)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 131072)];
    weight_shared[(((int)threadIdx.x) + 2176)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 132096)];
    weight_shared[(((int)threadIdx.x) + 2304)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 133120)];
    weight_shared[(((int)threadIdx.x) + 2432)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 134144)];
    weight_shared[(((int)threadIdx.x) + 2560)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 163840)];
    weight_shared[(((int)threadIdx.x) + 2688)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 164864)];
    weight_shared[(((int)threadIdx.x) + 2816)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 165888)];
    weight_shared[(((int)threadIdx.x) + 2944)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 166912)];
    weight_shared[(((int)threadIdx.x) + 3072)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 196608)];
    weight_shared[(((int)threadIdx.x) + 3200)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 197632)];
    weight_shared[(((int)threadIdx.x) + 3328)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 198656)];
    weight_shared[(((int)threadIdx.x) + 3456)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 199680)];
    weight_shared[(((int)threadIdx.x) + 3584)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 229376)];
    weight_shared[(((int)threadIdx.x) + 3712)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 230400)];
    weight_shared[(((int)threadIdx.x) + 3840)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 231424)];
    weight_shared[(((int)threadIdx.x) + 3968)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 232448)];
    weight_shared[(((int)threadIdx.x) + 4096)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 262144)];
    weight_shared[(((int)threadIdx.x) + 4224)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 263168)];
    weight_shared[(((int)threadIdx.x) + 4352)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 264192)];
    weight_shared[(((int)threadIdx.x) + 4480)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 265216)];
    weight_shared[(((int)threadIdx.x) + 4608)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 294912)];
    weight_shared[(((int)threadIdx.x) + 4736)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 295936)];
    weight_shared[(((int)threadIdx.x) + 4864)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 296960)];
    weight_shared[(((int)threadIdx.x) + 4992)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 297984)];
    weight_shared[(((int)threadIdx.x) + 5120)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 327680)];
    weight_shared[(((int)threadIdx.x) + 5248)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 328704)];
    weight_shared[(((int)threadIdx.x) + 5376)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 329728)];
    weight_shared[(((int)threadIdx.x) + 5504)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 330752)];
    weight_shared[(((int)threadIdx.x) + 5632)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 360448)];
    weight_shared[(((int)threadIdx.x) + 5760)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 361472)];
    weight_shared[(((int)threadIdx.x) + 5888)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 362496)];
    weight_shared[(((int)threadIdx.x) + 6016)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 363520)];
    weight_shared[(((int)threadIdx.x) + 6144)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 393216)];
    weight_shared[(((int)threadIdx.x) + 6272)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 394240)];
    weight_shared[(((int)threadIdx.x) + 6400)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 395264)];
    weight_shared[(((int)threadIdx.x) + 6528)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 396288)];
    weight_shared[(((int)threadIdx.x) + 6656)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 425984)];
    weight_shared[(((int)threadIdx.x) + 6784)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 427008)];
    weight_shared[(((int)threadIdx.x) + 6912)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 428032)];
    weight_shared[(((int)threadIdx.x) + 7040)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 429056)];
    weight_shared[(((int)threadIdx.x) + 7168)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 458752)];
    weight_shared[(((int)threadIdx.x) + 7296)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 459776)];
    weight_shared[(((int)threadIdx.x) + 7424)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 460800)];
    weight_shared[(((int)threadIdx.x) + 7552)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 461824)];
    weight_shared[(((int)threadIdx.x) + 7680)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 491520)];
    weight_shared[(((int)threadIdx.x) + 7808)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 492544)];
    weight_shared[(((int)threadIdx.x) + 7936)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 493568)];
    weight_shared[(((int)threadIdx.x) + 8064)] = weight[(((((rc_0 * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 494592)];
    __syncthreads();
    for (int rc_1 = 0; rc_1 < 32; ++rc_1) {
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_1)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 7680)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 32)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 6656)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 320)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 3584)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 352)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 2560)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 32)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 7168)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 64)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 6144)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 352)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 3072)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 384)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 2048)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 320)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 5632)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 352)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 4608)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 640)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 1536)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 672)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 512)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 352)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 5120)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 384)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 4096)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 672)] * weight_shared[(((rc_1 * 16) + (((int)threadIdx.x) & 15)) + 1024)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_1) + 704)] * weight_shared[((rc_1 * 16) + (((int)threadIdx.x) & 15))]));
    }
  }
  conv2d_transpose_nhwc[(((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[((((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 128)] = conv2d_transpose_nhwc_local[1];
  conv2d_transpose_nhwc[((((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 2048)] = conv2d_transpose_nhwc_local[2];
  conv2d_transpose_nhwc[((((((((int)blockIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 2176)] = conv2d_transpose_nhwc_local[3];
}

