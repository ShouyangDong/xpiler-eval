
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
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[8];
  __shared__ float PadInput_shared[3456];
  __shared__ float placeholder_shared[288];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[7] = 0.000000e+00f;
  float condval;
  if (((7 <= ((int)blockIdx.x)) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5))))) {
    condval = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3616)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  float condval_1;
  if ((7 <= ((int)blockIdx.x))) {
    condval_1 = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3360)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 256)] = condval_1;
  float condval_2;
  if ((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 512) / 576))) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)) < 113))) {
    condval_2 = placeholder[(((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 512) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((((((int)threadIdx.x) >> 5) + 16) % 18) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 512)] = condval_2;
  PadInput_shared[(((int)threadIdx.x) + 768)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 768) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3424)];
  float condval_3;
  if (((1 <= (((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 14) % 18))) && ((((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 14) % 18)) < 113))) {
    condval_3 = placeholder[(((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1024) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((((((int)threadIdx.x) >> 5) + 14) % 18) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1024)] = condval_3;
  PadInput_shared[(((int)threadIdx.x) + 1280)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1280) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3488)];
  float condval_4;
  if (((1 <= (((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 12) % 18))) && ((((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 12) % 18)) < 113))) {
    condval_4 = placeholder[(((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1536) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((((((int)threadIdx.x) >> 5) + 12) % 18) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1536)] = condval_4;
  PadInput_shared[(((int)threadIdx.x) + 1792)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 1792) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3552)];
  float condval_5;
  if (((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)) < 103)) {
    condval_5 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2048) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3296)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2048)] = condval_5;
  float condval_6;
  if ((1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)))) {
    condval_6 = placeholder[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) + 10720)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2304)] = condval_6;
  PadInput_shared[(((int)threadIdx.x) + 2560)] = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2560) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3360)];
  float condval_7;
  if (((((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 2816) / 576)) < 113) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + (((((int)threadIdx.x) >> 5) + 16) % 18)) < 113))) {
    condval_7 = placeholder[(((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 2816) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((((((int)threadIdx.x) >> 5) + 16) % 18) * 32)) + (((int)threadIdx.x) & 31)) - 3616)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 2816)] = condval_7;
  float condval_8;
  if (((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 3072) / 576)) < 113)) {
    condval_8 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 3072) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3424)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 3072)] = condval_8;
  if (((int)threadIdx.x) < 128) {
    float condval_9;
    if ((((((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) + 3328) / 576)) < 113) && ((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) >> 5)) < 99))) {
      condval_9 = placeholder[((((((((int)blockIdx.x) / 7) * 14336) + (((((int)threadIdx.x) + 3328) / 576) * 3584)) + ((((int)blockIdx.x) % 7) * 512)) + ((int)threadIdx.x)) - 3168)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 3328)] = condval_9;
  }
  if (((int)threadIdx.x) < 72) {
    *(float4*)(placeholder_shared + (((int)threadIdx.x) * 4)) = *(float4*)(placeholder_1 + (((int)threadIdx.x) * 4));
  }
  __syncthreads();
  for (int rh_1 = 0; rh_1 < 3; ++rh_1) {
    depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2))] * placeholder_shared[((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2))]));
    depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 256)] * placeholder_shared[((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2))]));
    depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 1)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 1)]));
    depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 257)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 1)]));
    depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 576)] * placeholder_shared[((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2))]));
    depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 832)] * placeholder_shared[((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2))]));
    depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 577)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 1)]));
    depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 833)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 1)]));
    depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 32)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 32)]));
    depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 288)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 32)]));
    depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 33)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 33)]));
    depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 289)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 33)]));
    depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 608)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 32)]));
    depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 864)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 32)]));
    depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 609)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 33)]));
    depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 865)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 33)]));
    depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 64)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 64)]));
    depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 320)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 64)]));
    depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 65)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 65)]));
    depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 321)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 65)]));
    depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 640)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 64)]));
    depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 896)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 64)]));
    depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 641)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 65)]));
    depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1152) + (rh_1 * 576)) + ((((int)threadIdx.x) & 127) * 2)) + 897)] * placeholder_shared[(((rh_1 * 96) + ((((int)threadIdx.x) & 15) * 2)) + 65)]));
  }
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2)) + 256)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2)) + 1)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2)) + 257)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2)) + 3584)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2)) + 3840)] = depth_conv2d_nhwc_local[6];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2)) + 3585)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 7) * 14336) + ((((int)threadIdx.x) >> 7) * 7168)) + ((((int)blockIdx.x) % 7) * 512)) + ((((int)threadIdx.x) & 127) * 2)) + 3841)] = depth_conv2d_nhwc_local[7];
}

