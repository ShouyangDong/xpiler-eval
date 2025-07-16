
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
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_transpose_nhwc_local[4];
  __shared__ float PadInput_shared[1536];
  __shared__ float weight_shared[8192];
  conv2d_transpose_nhwc_local[0] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[1] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[2] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[3] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 8; ++rc_0) {
    __syncthreads();
    float condval;
    if ((((((int)blockIdx.x) >> 5) == 1) && (32 <= ((int)threadIdx.x)))) {
      condval = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 2560)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) * 2)] = condval;
    float condval_1;
    if ((((((int)blockIdx.x) >> 5) == 1) && (32 <= ((int)threadIdx.x)))) {
      condval_1 = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 2559)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1)] = condval_1;
    float condval_2;
    if (((((int)blockIdx.x) >> 5) == 1)) {
      condval_2 = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 1536)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 128)] = condval_2;
    float condval_3;
    if (((((int)blockIdx.x) >> 5) == 1)) {
      condval_3 = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 1535)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 129)] = condval_3;
    float condval_4;
    if ((((((int)blockIdx.x) >> 5) == 1) && (((int)threadIdx.x) < 32))) {
      condval_4 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + (((int)threadIdx.x) * 2)) - 512)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 256)] = condval_4;
    float condval_5;
    if ((((((int)blockIdx.x) >> 5) == 1) && (((int)threadIdx.x) < 32))) {
      condval_5 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + (((int)threadIdx.x) * 2)) - 511)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 257)] = condval_5;
    float condval_6;
    if ((32 <= ((int)threadIdx.x))) {
      condval_6 = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 512)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 384)] = condval_6;
    float condval_7;
    if ((32 <= ((int)threadIdx.x))) {
      condval_7 = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 511)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 385)] = condval_7;
    PadInput_shared[((((int)threadIdx.x) * 2) + 512)] = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 512)];
    PadInput_shared[((((int)threadIdx.x) * 2) + 513)] = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 513)];
    float condval_8;
    if ((((int)threadIdx.x) < 32)) {
      condval_8 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + (((int)threadIdx.x) * 2)) + 1536)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 640)] = condval_8;
    float condval_9;
    if ((((int)threadIdx.x) < 32)) {
      condval_9 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + (((int)threadIdx.x) * 2)) + 1537)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 641)] = condval_9;
    float condval_10;
    if ((32 <= ((int)threadIdx.x))) {
      condval_10 = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1536)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 768)] = condval_10;
    float condval_11;
    if ((32 <= ((int)threadIdx.x))) {
      condval_11 = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1537)];
    } else {
      condval_11 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 769)] = condval_11;
    PadInput_shared[((((int)threadIdx.x) * 2) + 896)] = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 2560)];
    PadInput_shared[((((int)threadIdx.x) * 2) + 897)] = inputs[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 2561)];
    float condval_12;
    if ((((int)threadIdx.x) < 32)) {
      condval_12 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + (((int)threadIdx.x) * 2)) + 3584)];
    } else {
      condval_12 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1024)] = condval_12;
    float condval_13;
    if ((((int)threadIdx.x) < 32)) {
      condval_13 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + (((int)threadIdx.x) * 2)) + 3585)];
    } else {
      condval_13 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1025)] = condval_13;
    float condval_14;
    if (((((int)blockIdx.x) < 32) && (32 <= ((int)threadIdx.x)))) {
      condval_14 = inputs[(((((((int)threadIdx.x) >> 5) * 512) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 3584)];
    } else {
      condval_14 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1152)] = condval_14;
    float condval_15;
    if (((((int)blockIdx.x) < 32) && (32 <= ((int)threadIdx.x)))) {
      condval_15 = inputs[(((((((int)threadIdx.x) >> 5) * 512) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 3585)];
    } else {
      condval_15 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1153)] = condval_15;
    float condval_16;
    if ((((int)blockIdx.x) < 32)) {
      condval_16 = inputs[(((((((int)threadIdx.x) >> 5) * 512) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 4608)];
    } else {
      condval_16 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1280)] = condval_16;
    float condval_17;
    if ((((int)blockIdx.x) < 32)) {
      condval_17 = inputs[(((((((int)threadIdx.x) >> 5) * 512) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 4609)];
    } else {
      condval_17 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1281)] = condval_17;
    float condval_18;
    if (((((int)blockIdx.x) < 32) && (((int)threadIdx.x) < 32))) {
      condval_18 = inputs[(((rc_0 * 64) + (((int)threadIdx.x) * 2)) + 5632)];
    } else {
      condval_18 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1408)] = condval_18;
    float condval_19;
    if (((((int)blockIdx.x) < 32) && (((int)threadIdx.x) < 32))) {
      condval_19 = inputs[(((rc_0 * 64) + (((int)threadIdx.x) * 2)) + 5633)];
    } else {
      condval_19 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1409)] = condval_19;
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8192));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 131072));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 139264));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 262144));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 270336));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 393216));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 401408));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 524288));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 532480));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 655360));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 663552));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 786432));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 794624));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 917504));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 925696));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1048576));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4352)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1056768));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4608)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1179648));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4864)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1187840));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5120)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1310720));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5376)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1318912));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5632)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1441792));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5888)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1449984));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6144)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1572864));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6400)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1581056));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6656)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1703936));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6912)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1712128));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7168)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1835008));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7424)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1843200));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7680)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1966080));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7936)) = *(float4*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1974272));
    __syncthreads();
    for (int rh_1 = 0; rh_1 < 2; ++rh_1) {
      for (int rw_1 = 0; rw_1 < 2; ++rw_1) {
        for (int rc_1 = 0; rc_1 < 4; ++rc_1) {
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16))] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7680) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 1)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7688) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 2)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7696) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 3)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7704) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 4)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7712) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 5)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7720) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 6)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7728) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 7)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7736) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 8)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7744) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 9)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7752) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 10)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7760) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 11)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7768) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 12)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7776) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 13)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7784) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 14)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7792) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 15)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7800) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 64)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7168) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 65)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7176) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 66)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7184) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 67)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7192) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 68)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7200) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 69)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7208) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 70)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7216) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 71)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7224) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 72)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7232) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 73)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7240) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 74)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7248) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 75)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7256) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 76)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7264) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 77)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7272) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 78)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7280) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 79)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7288) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 384)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5632) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 385)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5640) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 386)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5648) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 387)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5656) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 388)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5664) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 389)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5672) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 390)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5680) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 391)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5688) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 392)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5696) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 393)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5704) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 394)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5712) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 395)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5720) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 396)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5728) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 397)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5736) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 398)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5744) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 399)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5752) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 448)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5120) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 449)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5128) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 450)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5136) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 451)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5144) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 452)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5152) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 453)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5160) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 454)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5168) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 455)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5176) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 456)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5184) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 457)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5192) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 458)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5200) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 459)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5208) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 460)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5216) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 461)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5224) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 462)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5232) - (rw_1 * 1024)) - (rh_1 * 4096))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rw_1 * 64)) + (rc_1 * 16)) + 463)] * weight_shared[(((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5240) - (rw_1 * 1024)) - (rh_1 * 4096))]));
        }
      }
    }
  }
  conv2d_transpose_nhwc[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7)) + 256)] = conv2d_transpose_nhwc_local[1];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7)) + 2048)] = conv2d_transpose_nhwc_local[2];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7)) + 2304)] = conv2d_transpose_nhwc_local[3];
}

