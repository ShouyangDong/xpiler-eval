
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_transpose_nhwc_local[64];
  __shared__ float PadInput_shared[576];
  __shared__ float weight_shared[2048];
  conv2d_transpose_nhwc_local[0] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[4] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[8] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[12] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[16] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[20] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[24] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[28] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[32] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[36] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[40] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[44] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[48] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[52] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[56] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[60] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[1] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[5] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[9] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[13] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[17] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[21] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[25] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[29] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[33] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[37] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[41] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[45] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[49] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[53] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[57] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[61] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[2] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[6] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[10] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[14] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[18] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[22] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[26] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[30] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[34] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[38] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[42] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[46] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[50] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[54] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[58] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[62] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[3] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[7] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[11] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[15] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[19] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[23] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[27] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[31] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[35] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[39] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[43] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[47] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[51] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[55] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[59] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[63] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 16; ++rc_0) {
    __syncthreads();
    PadInput_shared[((int)threadIdx.x)] = 0.000000e+00f;
    float condval;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 128)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 32)] = condval;
    float condval_1;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_1 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 1920)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 64)] = condval_1;
    float condval_2;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_2 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 3968)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 96)] = condval_2;
    float condval_3;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_3 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 6016)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 128)] = condval_3;
    float condval_4;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_4 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 8064)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 160)] = condval_4;
    float condval_5;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_5 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 10112)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 192)] = condval_5;
    float condval_6;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_6 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 12160)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 224)] = condval_6;
    float condval_7;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_7 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 14208)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 256)] = condval_7;
    float condval_8;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_8 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 16256)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 288)] = condval_8;
    float condval_9;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_9 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 18304)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 320)] = condval_9;
    float condval_10;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_10 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 20352)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 352)] = condval_10;
    float condval_11;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_11 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 22400)];
    } else {
      condval_11 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 384)] = condval_11;
    float condval_12;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_12 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 24448)];
    } else {
      condval_12 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 416)] = condval_12;
    float condval_13;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_13 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 26496)];
    } else {
      condval_13 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 448)] = condval_13;
    float condval_14;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_14 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 28544)];
    } else {
      condval_14 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 480)] = condval_14;
    float condval_15;
    if (((1 <= ((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3))) && (((((((int)blockIdx.x) & 31) >> 2) * 2) + (((int)threadIdx.x) >> 3)) < 17))) {
      condval_15 = inputs[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 3) * 128)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 30592)];
    } else {
      condval_15 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 512)] = condval_15;
    PadInput_shared[(((int)threadIdx.x) + 544)] = 0.000000e+00f;
    weight_shared[((int)threadIdx.x)] = weight[((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15))];
    weight_shared[(((int)threadIdx.x) + 32)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 128)];
    weight_shared[(((int)threadIdx.x) + 64)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 256)];
    weight_shared[(((int)threadIdx.x) + 96)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 384)];
    weight_shared[(((int)threadIdx.x) + 128)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 8192)];
    weight_shared[(((int)threadIdx.x) + 160)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 8320)];
    weight_shared[(((int)threadIdx.x) + 192)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 8448)];
    weight_shared[(((int)threadIdx.x) + 224)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 8576)];
    weight_shared[(((int)threadIdx.x) + 256)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 16384)];
    weight_shared[(((int)threadIdx.x) + 288)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 16512)];
    weight_shared[(((int)threadIdx.x) + 320)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 16640)];
    weight_shared[(((int)threadIdx.x) + 352)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 16768)];
    weight_shared[(((int)threadIdx.x) + 384)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 24576)];
    weight_shared[(((int)threadIdx.x) + 416)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 24704)];
    weight_shared[(((int)threadIdx.x) + 448)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 24832)];
    weight_shared[(((int)threadIdx.x) + 480)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 24960)];
    weight_shared[(((int)threadIdx.x) + 512)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 32768)];
    weight_shared[(((int)threadIdx.x) + 544)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 32896)];
    weight_shared[(((int)threadIdx.x) + 576)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 33024)];
    weight_shared[(((int)threadIdx.x) + 608)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 33152)];
    weight_shared[(((int)threadIdx.x) + 640)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 40960)];
    weight_shared[(((int)threadIdx.x) + 672)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 41088)];
    weight_shared[(((int)threadIdx.x) + 704)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 41216)];
    weight_shared[(((int)threadIdx.x) + 736)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 41344)];
    weight_shared[(((int)threadIdx.x) + 768)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 49152)];
    weight_shared[(((int)threadIdx.x) + 800)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 49280)];
    weight_shared[(((int)threadIdx.x) + 832)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 49408)];
    weight_shared[(((int)threadIdx.x) + 864)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 49536)];
    weight_shared[(((int)threadIdx.x) + 896)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 57344)];
    weight_shared[(((int)threadIdx.x) + 928)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 57472)];
    weight_shared[(((int)threadIdx.x) + 960)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 57600)];
    weight_shared[(((int)threadIdx.x) + 992)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 57728)];
    weight_shared[(((int)threadIdx.x) + 1024)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 65536)];
    weight_shared[(((int)threadIdx.x) + 1056)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 65664)];
    weight_shared[(((int)threadIdx.x) + 1088)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 65792)];
    weight_shared[(((int)threadIdx.x) + 1120)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 65920)];
    weight_shared[(((int)threadIdx.x) + 1152)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 73728)];
    weight_shared[(((int)threadIdx.x) + 1184)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 73856)];
    weight_shared[(((int)threadIdx.x) + 1216)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 73984)];
    weight_shared[(((int)threadIdx.x) + 1248)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 74112)];
    weight_shared[(((int)threadIdx.x) + 1280)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 81920)];
    weight_shared[(((int)threadIdx.x) + 1312)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 82048)];
    weight_shared[(((int)threadIdx.x) + 1344)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 82176)];
    weight_shared[(((int)threadIdx.x) + 1376)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 82304)];
    weight_shared[(((int)threadIdx.x) + 1408)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 90112)];
    weight_shared[(((int)threadIdx.x) + 1440)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 90240)];
    weight_shared[(((int)threadIdx.x) + 1472)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 90368)];
    weight_shared[(((int)threadIdx.x) + 1504)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 90496)];
    weight_shared[(((int)threadIdx.x) + 1536)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 98304)];
    weight_shared[(((int)threadIdx.x) + 1568)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 98432)];
    weight_shared[(((int)threadIdx.x) + 1600)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 98560)];
    weight_shared[(((int)threadIdx.x) + 1632)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 98688)];
    weight_shared[(((int)threadIdx.x) + 1664)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 106496)];
    weight_shared[(((int)threadIdx.x) + 1696)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 106624)];
    weight_shared[(((int)threadIdx.x) + 1728)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 106752)];
    weight_shared[(((int)threadIdx.x) + 1760)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 106880)];
    weight_shared[(((int)threadIdx.x) + 1792)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 114688)];
    weight_shared[(((int)threadIdx.x) + 1824)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 114816)];
    weight_shared[(((int)threadIdx.x) + 1856)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 114944)];
    weight_shared[(((int)threadIdx.x) + 1888)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 115072)];
    weight_shared[(((int)threadIdx.x) + 1920)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 122880)];
    weight_shared[(((int)threadIdx.x) + 1952)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 123008)];
    weight_shared[(((int)threadIdx.x) + 1984)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 123136)];
    weight_shared[(((int)threadIdx.x) + 2016)] = weight[(((((rc_0 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 123264)];
    __syncthreads();
    for (int rh_1 = 0; rh_1 < 2; ++rh_1) {
      for (int rw_1 = 0; rw_1 < 2; ++rw_1) {
        for (int rc_1 = 0; rc_1 < 8; ++rc_1) {
          conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1920) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1924) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[8] = (conv2d_transpose_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1928) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[12] = (conv2d_transpose_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1932) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[16] = (conv2d_transpose_nhwc_local[16] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1920) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[20] = (conv2d_transpose_nhwc_local[20] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1924) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[24] = (conv2d_transpose_nhwc_local[24] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1928) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[28] = (conv2d_transpose_nhwc_local[28] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1932) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[32] = (conv2d_transpose_nhwc_local[32] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 256)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1920) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[36] = (conv2d_transpose_nhwc_local[36] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 256)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1924) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[40] = (conv2d_transpose_nhwc_local[40] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 256)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1928) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[44] = (conv2d_transpose_nhwc_local[44] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 256)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1932) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[48] = (conv2d_transpose_nhwc_local[48] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1920) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[52] = (conv2d_transpose_nhwc_local[52] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1924) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[56] = (conv2d_transpose_nhwc_local[56] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1928) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[60] = (conv2d_transpose_nhwc_local[60] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1932) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1792) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1796) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[9] = (conv2d_transpose_nhwc_local[9] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1800) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[13] = (conv2d_transpose_nhwc_local[13] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 8)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1804) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[17] = (conv2d_transpose_nhwc_local[17] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 16)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1792) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[21] = (conv2d_transpose_nhwc_local[21] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 16)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1796) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[25] = (conv2d_transpose_nhwc_local[25] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 16)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1800) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[29] = (conv2d_transpose_nhwc_local[29] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 16)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1804) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[33] = (conv2d_transpose_nhwc_local[33] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1792) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[37] = (conv2d_transpose_nhwc_local[37] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1796) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[41] = (conv2d_transpose_nhwc_local[41] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1800) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[45] = (conv2d_transpose_nhwc_local[45] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 264)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1804) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[49] = (conv2d_transpose_nhwc_local[49] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 272)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1792) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[53] = (conv2d_transpose_nhwc_local[53] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 272)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1796) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[57] = (conv2d_transpose_nhwc_local[57] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 272)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1800) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[61] = (conv2d_transpose_nhwc_local[61] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 272)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1804) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 32)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1408) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 32)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1412) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[10] = (conv2d_transpose_nhwc_local[10] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 32)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1416) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[14] = (conv2d_transpose_nhwc_local[14] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 32)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1420) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[18] = (conv2d_transpose_nhwc_local[18] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1408) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[22] = (conv2d_transpose_nhwc_local[22] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1412) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[26] = (conv2d_transpose_nhwc_local[26] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1416) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[30] = (conv2d_transpose_nhwc_local[30] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1420) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[34] = (conv2d_transpose_nhwc_local[34] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 288)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1408) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[38] = (conv2d_transpose_nhwc_local[38] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 288)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1412) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[42] = (conv2d_transpose_nhwc_local[42] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 288)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1416) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[46] = (conv2d_transpose_nhwc_local[46] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 288)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1420) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[50] = (conv2d_transpose_nhwc_local[50] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1408) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[54] = (conv2d_transpose_nhwc_local[54] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1412) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[58] = (conv2d_transpose_nhwc_local[58] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1416) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[62] = (conv2d_transpose_nhwc_local[62] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1420) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1280) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1284) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[11] = (conv2d_transpose_nhwc_local[11] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1288) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[15] = (conv2d_transpose_nhwc_local[15] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 40)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1292) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[19] = (conv2d_transpose_nhwc_local[19] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 48)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1280) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[23] = (conv2d_transpose_nhwc_local[23] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 48)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1284) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[27] = (conv2d_transpose_nhwc_local[27] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 48)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1288) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[31] = (conv2d_transpose_nhwc_local[31] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 48)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1292) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[35] = (conv2d_transpose_nhwc_local[35] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1280) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[39] = (conv2d_transpose_nhwc_local[39] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1284) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[43] = (conv2d_transpose_nhwc_local[43] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1288) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[47] = (conv2d_transpose_nhwc_local[47] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 296)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1292) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[51] = (conv2d_transpose_nhwc_local[51] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 304)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1280) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[55] = (conv2d_transpose_nhwc_local[55] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 304)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1284) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[59] = (conv2d_transpose_nhwc_local[59] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 304)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1288) - (rw_1 * 256)) - (rh_1 * 1024))]));
          conv2d_transpose_nhwc_local[63] = (conv2d_transpose_nhwc_local[63] + (PadInput_shared[((((((((int)threadIdx.x) >> 2) * 32) + (rh_1 * 32)) + (rw_1 * 8)) + rc_1) + 304)] * weight_shared[(((((rc_1 * 16) + (((int)threadIdx.x) & 3)) + 1292) - (rw_1 * 256)) - (rh_1 * 1024))]));
        }
      }
    }
  }
  conv2d_transpose_nhwc[((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 4)] = conv2d_transpose_nhwc_local[4];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 8)] = conv2d_transpose_nhwc_local[8];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 12)] = conv2d_transpose_nhwc_local[12];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 128)] = conv2d_transpose_nhwc_local[16];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 132)] = conv2d_transpose_nhwc_local[20];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 136)] = conv2d_transpose_nhwc_local[24];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 140)] = conv2d_transpose_nhwc_local[28];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32768)] = conv2d_transpose_nhwc_local[32];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32772)] = conv2d_transpose_nhwc_local[36];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32776)] = conv2d_transpose_nhwc_local[40];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32780)] = conv2d_transpose_nhwc_local[44];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32896)] = conv2d_transpose_nhwc_local[48];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32900)] = conv2d_transpose_nhwc_local[52];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32904)] = conv2d_transpose_nhwc_local[56];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32908)] = conv2d_transpose_nhwc_local[60];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 64)] = conv2d_transpose_nhwc_local[1];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 68)] = conv2d_transpose_nhwc_local[5];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 72)] = conv2d_transpose_nhwc_local[9];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 76)] = conv2d_transpose_nhwc_local[13];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 192)] = conv2d_transpose_nhwc_local[17];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 196)] = conv2d_transpose_nhwc_local[21];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 200)] = conv2d_transpose_nhwc_local[25];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 204)] = conv2d_transpose_nhwc_local[29];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32832)] = conv2d_transpose_nhwc_local[33];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32836)] = conv2d_transpose_nhwc_local[37];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32840)] = conv2d_transpose_nhwc_local[41];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32844)] = conv2d_transpose_nhwc_local[45];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32960)] = conv2d_transpose_nhwc_local[49];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32964)] = conv2d_transpose_nhwc_local[53];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32968)] = conv2d_transpose_nhwc_local[57];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 32972)] = conv2d_transpose_nhwc_local[61];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2048)] = conv2d_transpose_nhwc_local[2];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2052)] = conv2d_transpose_nhwc_local[6];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2056)] = conv2d_transpose_nhwc_local[10];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2060)] = conv2d_transpose_nhwc_local[14];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2176)] = conv2d_transpose_nhwc_local[18];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2180)] = conv2d_transpose_nhwc_local[22];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2184)] = conv2d_transpose_nhwc_local[26];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2188)] = conv2d_transpose_nhwc_local[30];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34816)] = conv2d_transpose_nhwc_local[34];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34820)] = conv2d_transpose_nhwc_local[38];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34824)] = conv2d_transpose_nhwc_local[42];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34828)] = conv2d_transpose_nhwc_local[46];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34944)] = conv2d_transpose_nhwc_local[50];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34948)] = conv2d_transpose_nhwc_local[54];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34952)] = conv2d_transpose_nhwc_local[58];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34956)] = conv2d_transpose_nhwc_local[62];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2112)] = conv2d_transpose_nhwc_local[3];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2116)] = conv2d_transpose_nhwc_local[7];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2120)] = conv2d_transpose_nhwc_local[11];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2124)] = conv2d_transpose_nhwc_local[15];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2240)] = conv2d_transpose_nhwc_local[19];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2244)] = conv2d_transpose_nhwc_local[23];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2248)] = conv2d_transpose_nhwc_local[27];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 2252)] = conv2d_transpose_nhwc_local[31];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34880)] = conv2d_transpose_nhwc_local[35];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34884)] = conv2d_transpose_nhwc_local[39];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34888)] = conv2d_transpose_nhwc_local[43];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 34892)] = conv2d_transpose_nhwc_local[47];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 35008)] = conv2d_transpose_nhwc_local[51];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 35012)] = conv2d_transpose_nhwc_local[55];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 35016)] = conv2d_transpose_nhwc_local[59];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 2) * 4096)) + (((((int)blockIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 3)) + 35020)] = conv2d_transpose_nhwc_local[63];
}

