
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
extern "C" __global__ void __launch_bounds__(96) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(96) main_kernel(float* __restrict__ conv2d_transpose_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_transpose_nhwc_local[4];
  __shared__ float PadInput_shared[576];
  __shared__ float weight_shared[192];
  conv2d_transpose_nhwc_local[0] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[1] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[2] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[3] = 0.000000e+00f;
  for (int rw_0 = 0; rw_0 < 2; ++rw_0) {
    for (int rc_0 = 0; rc_0 < 8; ++rc_0) {
      __syncthreads();
      float condval;
      if ((((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 4) + (((int)threadIdx.x) / 24))) && (1 <= ((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0) < 33))) {
        condval = inputs[((((((((((((int)blockIdx.x) >> 7) * 262144) + (((((int)blockIdx.x) & 127) >> 4) * 8192)) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((((int)threadIdx.x) % 24) >> 3) * 64)) + (rw_0 * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2112)];
      } else {
        condval = 0.000000e+00f;
      }
      PadInput_shared[((int)threadIdx.x)] = condval;
      float condval_1;
      if (((((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 4) + (((((int)threadIdx.x) / 24) + 4) % 6))) && (((((((int)blockIdx.x) & 127) >> 4) * 4) + (((((int)threadIdx.x) / 24) + 4) % 6)) < 33)) && (1 <= ((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0) < 33))) {
        condval_1 = inputs[(((((((((((((int)blockIdx.x) >> 7) * 262144) + (((((int)threadIdx.x) + 96) / 144) * 65536)) + (((((int)blockIdx.x) & 127) >> 4) * 8192)) + ((((((int)threadIdx.x) / 24) + 4) % 6) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((((int)threadIdx.x) % 24) >> 3) * 64)) + (rw_0 * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2112)];
      } else {
        condval_1 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 96)] = condval_1;
      float condval_2;
      if ((((((((((int)blockIdx.x) & 127) >> 4) * 4) + (((int)threadIdx.x) / 24)) < 31) && (1 <= ((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0) < 33))) {
        condval_2 = inputs[(((((((((((((int)blockIdx.x) >> 7) * 262144) + (((((int)threadIdx.x) + 192) / 144) * 65536)) + (((((int)blockIdx.x) & 127) >> 4) * 8192)) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((((int)threadIdx.x) % 24) >> 3) * 64)) + (rw_0 * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 1984)];
      } else {
        condval_2 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 192)] = condval_2;
      float condval_3;
      if ((((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 4) + (((int)threadIdx.x) / 24))) && (1 <= ((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0) < 33))) {
        condval_3 = inputs[((((((((((((int)blockIdx.x) >> 7) * 262144) + (((((int)blockIdx.x) & 127) >> 4) * 8192)) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((((int)threadIdx.x) % 24) >> 3) * 64)) + (rw_0 * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 128960)];
      } else {
        condval_3 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 288)] = condval_3;
      float condval_4;
      if (((((1 <= ((((((int)blockIdx.x) & 127) >> 4) * 4) + (((((int)threadIdx.x) / 24) + 4) % 6))) && (((((((int)blockIdx.x) & 127) >> 4) * 4) + (((((int)threadIdx.x) / 24) + 4) % 6)) < 33)) && (1 <= ((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0) < 33))) {
        condval_4 = inputs[(((((((((((((int)blockIdx.x) >> 7) * 262144) + (((((int)threadIdx.x) + 384) / 144) * 65536)) + (((((int)blockIdx.x) & 127) >> 4) * 8192)) + ((((((int)threadIdx.x) / 24) + 4) % 6) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((((int)threadIdx.x) % 24) >> 3) * 64)) + (rw_0 * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 2112)];
      } else {
        condval_4 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 384)] = condval_4;
      float condval_5;
      if ((((((((((int)blockIdx.x) & 127) >> 4) * 4) + (((int)threadIdx.x) / 24)) < 31) && (1 <= ((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0))) && (((((((int)blockIdx.x) & 15) * 2) + ((((int)threadIdx.x) % 24) >> 3)) + rw_0) < 33))) {
        condval_5 = inputs[(((((((((((((int)blockIdx.x) >> 7) * 262144) + (((((int)threadIdx.x) + 480) / 144) * 65536)) + (((((int)blockIdx.x) & 127) >> 4) * 8192)) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((((int)threadIdx.x) % 24) >> 3) * 64)) + (rw_0 * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 1984)];
      } else {
        condval_5 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 480)] = condval_5;
      weight_shared[((int)threadIdx.x)] = weight[(((((((((int)threadIdx.x) / 48) * 768) + (((((int)threadIdx.x) % 48) / 24) * 192)) + (rc_0 * 24)) + (((int)threadIdx.x) % 24)) + 384) - (rw_0 * 384))];
      weight_shared[(((int)threadIdx.x) + 96)] = weight[(((((((((int)threadIdx.x) / 48) * 768) + (((((int)threadIdx.x) % 48) / 24) * 192)) + (rc_0 * 24)) + (((int)threadIdx.x) % 24)) + 1920) - (rw_0 * 384))];
      __syncthreads();
      float condval_6;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_6 = PadInput_shared[((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8))];
      } else {
        condval_6 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_6 * weight_shared[((((int)threadIdx.x) % 3) + 168)]));
      float condval_7;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_7 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 24)];
      } else {
        condval_7 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_7 * weight_shared[((((int)threadIdx.x) % 3) + 120)]));
      float condval_8;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_8 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 24)];
      } else {
        condval_8 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_8 * weight_shared[((((int)threadIdx.x) % 3) + 168)]));
      float condval_9;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_9 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 48)];
      } else {
        condval_9 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_9 * weight_shared[((((int)threadIdx.x) % 3) + 120)]));
      float condval_10;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_10 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 1)];
      } else {
        condval_10 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_10 * weight_shared[((((int)threadIdx.x) % 3) + 171)]));
      float condval_11;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_11 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 25)];
      } else {
        condval_11 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_11 * weight_shared[((((int)threadIdx.x) % 3) + 123)]));
      float condval_12;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_12 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 25)];
      } else {
        condval_12 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_12 * weight_shared[((((int)threadIdx.x) % 3) + 171)]));
      float condval_13;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_13 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 49)];
      } else {
        condval_13 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_13 * weight_shared[((((int)threadIdx.x) % 3) + 123)]));
      float condval_14;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_14 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 2)];
      } else {
        condval_14 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_14 * weight_shared[((((int)threadIdx.x) % 3) + 174)]));
      float condval_15;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_15 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 26)];
      } else {
        condval_15 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_15 * weight_shared[((((int)threadIdx.x) % 3) + 126)]));
      float condval_16;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_16 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 26)];
      } else {
        condval_16 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_16 * weight_shared[((((int)threadIdx.x) % 3) + 174)]));
      float condval_17;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_17 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 50)];
      } else {
        condval_17 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_17 * weight_shared[((((int)threadIdx.x) % 3) + 126)]));
      float condval_18;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_18 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 3)];
      } else {
        condval_18 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_18 * weight_shared[((((int)threadIdx.x) % 3) + 177)]));
      float condval_19;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_19 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 27)];
      } else {
        condval_19 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_19 * weight_shared[((((int)threadIdx.x) % 3) + 129)]));
      float condval_20;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_20 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 27)];
      } else {
        condval_20 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_20 * weight_shared[((((int)threadIdx.x) % 3) + 177)]));
      float condval_21;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_21 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 51)];
      } else {
        condval_21 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_21 * weight_shared[((((int)threadIdx.x) % 3) + 129)]));
      float condval_22;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_22 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 4)];
      } else {
        condval_22 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_22 * weight_shared[((((int)threadIdx.x) % 3) + 180)]));
      float condval_23;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_23 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 28)];
      } else {
        condval_23 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_23 * weight_shared[((((int)threadIdx.x) % 3) + 132)]));
      float condval_24;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_24 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 28)];
      } else {
        condval_24 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_24 * weight_shared[((((int)threadIdx.x) % 3) + 180)]));
      float condval_25;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_25 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 52)];
      } else {
        condval_25 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_25 * weight_shared[((((int)threadIdx.x) % 3) + 132)]));
      float condval_26;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_26 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 5)];
      } else {
        condval_26 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_26 * weight_shared[((((int)threadIdx.x) % 3) + 183)]));
      float condval_27;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_27 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 29)];
      } else {
        condval_27 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_27 * weight_shared[((((int)threadIdx.x) % 3) + 135)]));
      float condval_28;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_28 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 29)];
      } else {
        condval_28 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_28 * weight_shared[((((int)threadIdx.x) % 3) + 183)]));
      float condval_29;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_29 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 53)];
      } else {
        condval_29 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_29 * weight_shared[((((int)threadIdx.x) % 3) + 135)]));
      float condval_30;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_30 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 6)];
      } else {
        condval_30 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_30 * weight_shared[((((int)threadIdx.x) % 3) + 186)]));
      float condval_31;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_31 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 30)];
      } else {
        condval_31 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_31 * weight_shared[((((int)threadIdx.x) % 3) + 138)]));
      float condval_32;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_32 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 30)];
      } else {
        condval_32 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_32 * weight_shared[((((int)threadIdx.x) % 3) + 186)]));
      float condval_33;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_33 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 54)];
      } else {
        condval_33 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_33 * weight_shared[((((int)threadIdx.x) % 3) + 138)]));
      float condval_34;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_34 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 7)];
      } else {
        condval_34 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_34 * weight_shared[((((int)threadIdx.x) % 3) + 189)]));
      float condval_35;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_35 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 31)];
      } else {
        condval_35 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_35 * weight_shared[((((int)threadIdx.x) % 3) + 141)]));
      float condval_36;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_36 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 31)];
      } else {
        condval_36 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_36 * weight_shared[((((int)threadIdx.x) % 3) + 189)]));
      float condval_37;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_37 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 55)];
      } else {
        condval_37 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_37 * weight_shared[((((int)threadIdx.x) % 3) + 141)]));
      float condval_38;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_38 = PadInput_shared[((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8))];
      } else {
        condval_38 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_38 * weight_shared[((((int)threadIdx.x) % 3) + 144)]));
      float condval_39;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_39 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 24)];
      } else {
        condval_39 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_39 * weight_shared[((((int)threadIdx.x) % 3) + 96)]));
      float condval_40;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_40 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 24)];
      } else {
        condval_40 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_40 * weight_shared[((((int)threadIdx.x) % 3) + 144)]));
      float condval_41;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_41 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 48)];
      } else {
        condval_41 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_41 * weight_shared[((((int)threadIdx.x) % 3) + 96)]));
      float condval_42;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_42 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 1)];
      } else {
        condval_42 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_42 * weight_shared[((((int)threadIdx.x) % 3) + 147)]));
      float condval_43;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_43 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 25)];
      } else {
        condval_43 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_43 * weight_shared[((((int)threadIdx.x) % 3) + 99)]));
      float condval_44;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_44 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 25)];
      } else {
        condval_44 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_44 * weight_shared[((((int)threadIdx.x) % 3) + 147)]));
      float condval_45;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_45 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 49)];
      } else {
        condval_45 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_45 * weight_shared[((((int)threadIdx.x) % 3) + 99)]));
      float condval_46;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_46 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 2)];
      } else {
        condval_46 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_46 * weight_shared[((((int)threadIdx.x) % 3) + 150)]));
      float condval_47;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_47 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 26)];
      } else {
        condval_47 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_47 * weight_shared[((((int)threadIdx.x) % 3) + 102)]));
      float condval_48;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_48 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 26)];
      } else {
        condval_48 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_48 * weight_shared[((((int)threadIdx.x) % 3) + 150)]));
      float condval_49;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_49 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 50)];
      } else {
        condval_49 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_49 * weight_shared[((((int)threadIdx.x) % 3) + 102)]));
      float condval_50;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_50 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 3)];
      } else {
        condval_50 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_50 * weight_shared[((((int)threadIdx.x) % 3) + 153)]));
      float condval_51;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_51 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 27)];
      } else {
        condval_51 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_51 * weight_shared[((((int)threadIdx.x) % 3) + 105)]));
      float condval_52;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_52 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 27)];
      } else {
        condval_52 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_52 * weight_shared[((((int)threadIdx.x) % 3) + 153)]));
      float condval_53;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_53 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 51)];
      } else {
        condval_53 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_53 * weight_shared[((((int)threadIdx.x) % 3) + 105)]));
      float condval_54;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_54 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 4)];
      } else {
        condval_54 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_54 * weight_shared[((((int)threadIdx.x) % 3) + 156)]));
      float condval_55;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_55 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 28)];
      } else {
        condval_55 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_55 * weight_shared[((((int)threadIdx.x) % 3) + 108)]));
      float condval_56;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_56 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 28)];
      } else {
        condval_56 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_56 * weight_shared[((((int)threadIdx.x) % 3) + 156)]));
      float condval_57;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_57 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 52)];
      } else {
        condval_57 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_57 * weight_shared[((((int)threadIdx.x) % 3) + 108)]));
      float condval_58;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_58 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 5)];
      } else {
        condval_58 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_58 * weight_shared[((((int)threadIdx.x) % 3) + 159)]));
      float condval_59;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_59 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 29)];
      } else {
        condval_59 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_59 * weight_shared[((((int)threadIdx.x) % 3) + 111)]));
      float condval_60;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_60 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 29)];
      } else {
        condval_60 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_60 * weight_shared[((((int)threadIdx.x) % 3) + 159)]));
      float condval_61;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_61 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 53)];
      } else {
        condval_61 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_61 * weight_shared[((((int)threadIdx.x) % 3) + 111)]));
      float condval_62;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_62 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 6)];
      } else {
        condval_62 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_62 * weight_shared[((((int)threadIdx.x) % 3) + 162)]));
      float condval_63;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_63 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 30)];
      } else {
        condval_63 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_63 * weight_shared[((((int)threadIdx.x) % 3) + 114)]));
      float condval_64;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_64 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 30)];
      } else {
        condval_64 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_64 * weight_shared[((((int)threadIdx.x) % 3) + 162)]));
      float condval_65;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_65 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 54)];
      } else {
        condval_65 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_65 * weight_shared[((((int)threadIdx.x) % 3) + 114)]));
      float condval_66;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_66 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 7)];
      } else {
        condval_66 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_66 * weight_shared[((((int)threadIdx.x) % 3) + 165)]));
      float condval_67;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_67 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 31)];
      } else {
        condval_67 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_67 * weight_shared[((((int)threadIdx.x) % 3) + 117)]));
      float condval_68;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_68 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 31)];
      } else {
        condval_68 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_68 * weight_shared[((((int)threadIdx.x) % 3) + 165)]));
      float condval_69;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_69 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 55)];
      } else {
        condval_69 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_69 * weight_shared[((((int)threadIdx.x) % 3) + 117)]));
      float condval_70;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_70 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 24)];
      } else {
        condval_70 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_70 * weight_shared[((((int)threadIdx.x) % 3) + 72)]));
      float condval_71;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_71 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 48)];
      } else {
        condval_71 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_71 * weight_shared[((((int)threadIdx.x) % 3) + 24)]));
      float condval_72;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_72 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 48)];
      } else {
        condval_72 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_72 * weight_shared[((((int)threadIdx.x) % 3) + 72)]));
      float condval_73;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_73 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 72)];
      } else {
        condval_73 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_73 * weight_shared[((((int)threadIdx.x) % 3) + 24)]));
      float condval_74;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_74 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 25)];
      } else {
        condval_74 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_74 * weight_shared[((((int)threadIdx.x) % 3) + 75)]));
      float condval_75;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_75 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 49)];
      } else {
        condval_75 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_75 * weight_shared[((((int)threadIdx.x) % 3) + 27)]));
      float condval_76;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_76 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 49)];
      } else {
        condval_76 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_76 * weight_shared[((((int)threadIdx.x) % 3) + 75)]));
      float condval_77;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_77 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 73)];
      } else {
        condval_77 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_77 * weight_shared[((((int)threadIdx.x) % 3) + 27)]));
      float condval_78;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_78 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 26)];
      } else {
        condval_78 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_78 * weight_shared[((((int)threadIdx.x) % 3) + 78)]));
      float condval_79;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_79 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 50)];
      } else {
        condval_79 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_79 * weight_shared[((((int)threadIdx.x) % 3) + 30)]));
      float condval_80;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_80 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 50)];
      } else {
        condval_80 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_80 * weight_shared[((((int)threadIdx.x) % 3) + 78)]));
      float condval_81;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_81 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 74)];
      } else {
        condval_81 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_81 * weight_shared[((((int)threadIdx.x) % 3) + 30)]));
      float condval_82;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_82 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 27)];
      } else {
        condval_82 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_82 * weight_shared[((((int)threadIdx.x) % 3) + 81)]));
      float condval_83;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_83 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 51)];
      } else {
        condval_83 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_83 * weight_shared[((((int)threadIdx.x) % 3) + 33)]));
      float condval_84;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_84 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 51)];
      } else {
        condval_84 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_84 * weight_shared[((((int)threadIdx.x) % 3) + 81)]));
      float condval_85;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_85 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 75)];
      } else {
        condval_85 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_85 * weight_shared[((((int)threadIdx.x) % 3) + 33)]));
      float condval_86;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_86 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 28)];
      } else {
        condval_86 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_86 * weight_shared[((((int)threadIdx.x) % 3) + 84)]));
      float condval_87;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_87 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 52)];
      } else {
        condval_87 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_87 * weight_shared[((((int)threadIdx.x) % 3) + 36)]));
      float condval_88;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_88 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 52)];
      } else {
        condval_88 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_88 * weight_shared[((((int)threadIdx.x) % 3) + 84)]));
      float condval_89;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_89 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 76)];
      } else {
        condval_89 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_89 * weight_shared[((((int)threadIdx.x) % 3) + 36)]));
      float condval_90;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_90 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 29)];
      } else {
        condval_90 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_90 * weight_shared[((((int)threadIdx.x) % 3) + 87)]));
      float condval_91;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_91 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 53)];
      } else {
        condval_91 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_91 * weight_shared[((((int)threadIdx.x) % 3) + 39)]));
      float condval_92;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_92 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 53)];
      } else {
        condval_92 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_92 * weight_shared[((((int)threadIdx.x) % 3) + 87)]));
      float condval_93;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_93 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 77)];
      } else {
        condval_93 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_93 * weight_shared[((((int)threadIdx.x) % 3) + 39)]));
      float condval_94;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_94 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 30)];
      } else {
        condval_94 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_94 * weight_shared[((((int)threadIdx.x) % 3) + 90)]));
      float condval_95;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_95 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 54)];
      } else {
        condval_95 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_95 * weight_shared[((((int)threadIdx.x) % 3) + 42)]));
      float condval_96;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_96 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 54)];
      } else {
        condval_96 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_96 * weight_shared[((((int)threadIdx.x) % 3) + 90)]));
      float condval_97;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_97 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 78)];
      } else {
        condval_97 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_97 * weight_shared[((((int)threadIdx.x) % 3) + 42)]));
      float condval_98;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_98 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 31)];
      } else {
        condval_98 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_98 * weight_shared[((((int)threadIdx.x) % 3) + 93)]));
      float condval_99;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_99 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 55)];
      } else {
        condval_99 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_99 * weight_shared[((((int)threadIdx.x) % 3) + 45)]));
      float condval_100;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_100 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 55)];
      } else {
        condval_100 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_100 * weight_shared[((((int)threadIdx.x) % 3) + 93)]));
      float condval_101;
      if ((((((int)threadIdx.x) % 6) / 3) == 0)) {
        condval_101 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + (((((int)threadIdx.x) % 12) / 6) * 8)) + 79)];
      } else {
        condval_101 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_101 * weight_shared[((((int)threadIdx.x) % 3) + 45)]));
      float condval_102;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_102 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 24)];
      } else {
        condval_102 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_102 * weight_shared[((((int)threadIdx.x) % 3) + 48)]));
      float condval_103;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_103 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 48)];
      } else {
        condval_103 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_103 * weight_shared[(((int)threadIdx.x) % 3)]));
      float condval_104;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_104 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 48)];
      } else {
        condval_104 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_104 * weight_shared[((((int)threadIdx.x) % 3) + 48)]));
      float condval_105;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_105 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 72)];
      } else {
        condval_105 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_105 * weight_shared[(((int)threadIdx.x) % 3)]));
      float condval_106;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_106 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 25)];
      } else {
        condval_106 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_106 * weight_shared[((((int)threadIdx.x) % 3) + 51)]));
      float condval_107;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_107 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 49)];
      } else {
        condval_107 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_107 * weight_shared[((((int)threadIdx.x) % 3) + 3)]));
      float condval_108;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_108 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 49)];
      } else {
        condval_108 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_108 * weight_shared[((((int)threadIdx.x) % 3) + 51)]));
      float condval_109;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_109 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 73)];
      } else {
        condval_109 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_109 * weight_shared[((((int)threadIdx.x) % 3) + 3)]));
      float condval_110;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_110 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 26)];
      } else {
        condval_110 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_110 * weight_shared[((((int)threadIdx.x) % 3) + 54)]));
      float condval_111;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_111 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 50)];
      } else {
        condval_111 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_111 * weight_shared[((((int)threadIdx.x) % 3) + 6)]));
      float condval_112;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_112 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 50)];
      } else {
        condval_112 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_112 * weight_shared[((((int)threadIdx.x) % 3) + 54)]));
      float condval_113;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_113 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 74)];
      } else {
        condval_113 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_113 * weight_shared[((((int)threadIdx.x) % 3) + 6)]));
      float condval_114;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_114 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 27)];
      } else {
        condval_114 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_114 * weight_shared[((((int)threadIdx.x) % 3) + 57)]));
      float condval_115;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_115 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 51)];
      } else {
        condval_115 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_115 * weight_shared[((((int)threadIdx.x) % 3) + 9)]));
      float condval_116;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_116 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 51)];
      } else {
        condval_116 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_116 * weight_shared[((((int)threadIdx.x) % 3) + 57)]));
      float condval_117;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_117 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 75)];
      } else {
        condval_117 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_117 * weight_shared[((((int)threadIdx.x) % 3) + 9)]));
      float condval_118;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_118 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 28)];
      } else {
        condval_118 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_118 * weight_shared[((((int)threadIdx.x) % 3) + 60)]));
      float condval_119;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_119 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 52)];
      } else {
        condval_119 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_119 * weight_shared[((((int)threadIdx.x) % 3) + 12)]));
      float condval_120;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_120 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 52)];
      } else {
        condval_120 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_120 * weight_shared[((((int)threadIdx.x) % 3) + 60)]));
      float condval_121;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_121 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 76)];
      } else {
        condval_121 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_121 * weight_shared[((((int)threadIdx.x) % 3) + 12)]));
      float condval_122;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_122 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 29)];
      } else {
        condval_122 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_122 * weight_shared[((((int)threadIdx.x) % 3) + 63)]));
      float condval_123;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_123 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 53)];
      } else {
        condval_123 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_123 * weight_shared[((((int)threadIdx.x) % 3) + 15)]));
      float condval_124;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_124 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 53)];
      } else {
        condval_124 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_124 * weight_shared[((((int)threadIdx.x) % 3) + 63)]));
      float condval_125;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_125 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 77)];
      } else {
        condval_125 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_125 * weight_shared[((((int)threadIdx.x) % 3) + 15)]));
      float condval_126;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_126 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 30)];
      } else {
        condval_126 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_126 * weight_shared[((((int)threadIdx.x) % 3) + 66)]));
      float condval_127;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_127 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 54)];
      } else {
        condval_127 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_127 * weight_shared[((((int)threadIdx.x) % 3) + 18)]));
      float condval_128;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_128 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 54)];
      } else {
        condval_128 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_128 * weight_shared[((((int)threadIdx.x) % 3) + 66)]));
      float condval_129;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_129 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 78)];
      } else {
        condval_129 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_129 * weight_shared[((((int)threadIdx.x) % 3) + 18)]));
      float condval_130;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_130 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 31)];
      } else {
        condval_130 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_130 * weight_shared[((((int)threadIdx.x) % 3) + 69)]));
      float condval_131;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_131 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 55)];
      } else {
        condval_131 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_131 * weight_shared[((((int)threadIdx.x) % 3) + 21)]));
      float condval_132;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_132 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 55)];
      } else {
        condval_132 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (condval_132 * weight_shared[((((int)threadIdx.x) % 3) + 69)]));
      float condval_133;
      if ((((((((int)threadIdx.x) % 12) / 3) + 1) % 2) == 0)) {
        condval_133 = PadInput_shared[(((((((int)threadIdx.x) / 24) * 144) + (((((int)threadIdx.x) % 24) / 12) * 48)) + ((((((int)threadIdx.x) % 12) + 3) / 6) * 8)) + 79)];
      } else {
        condval_133 = 0.000000e+00f;
      }
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (condval_133 * weight_shared[((((int)threadIdx.x) % 3) + 21)]));
    }
  }
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 7) * 49152) + ((((int)threadIdx.x) / 24) * 12288)) + (((((int)blockIdx.x) & 127) >> 4) * 1536)) + (((((int)threadIdx.x) % 24) / 12) * 768)) + ((((int)blockIdx.x) & 15) * 12)) + (((int)threadIdx.x) % 12))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 7) * 49152) + ((((int)threadIdx.x) / 24) * 12288)) + (((((int)blockIdx.x) & 127) >> 4) * 1536)) + (((((int)threadIdx.x) % 24) / 12) * 768)) + ((((int)blockIdx.x) & 15) * 12)) + (((int)threadIdx.x) % 12)) + 192)] = conv2d_transpose_nhwc_local[1];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 7) * 49152) + ((((int)threadIdx.x) / 24) * 12288)) + (((((int)blockIdx.x) & 127) >> 4) * 1536)) + (((((int)threadIdx.x) % 24) / 12) * 768)) + ((((int)blockIdx.x) & 15) * 12)) + (((int)threadIdx.x) % 12)) + 384)] = conv2d_transpose_nhwc_local[2];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 7) * 49152) + ((((int)threadIdx.x) / 24) * 12288)) + (((((int)blockIdx.x) & 127) >> 4) * 1536)) + (((((int)threadIdx.x) % 24) / 12) * 768)) + ((((int)blockIdx.x) & 15) * 12)) + (((int)threadIdx.x) % 12)) + 576)] = conv2d_transpose_nhwc_local[3];
}

