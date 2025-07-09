
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
extern "C" __global__ void __launch_bounds__(192) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(192) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[2];
  __shared__ float PadInput_shared[3072];
  __shared__ float weight_shared[3072];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  for (int rh_0 = 0; rh_0 < 3; ++rh_0) {
    for (int rc_0 = 0; rc_0 < 4; ++rc_0) {
      __syncthreads();
      float condval;
      if (((1 < (((((int)blockIdx.x) / 48) * 6) + (rh_0 * 2))) && (1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5))))) {
        condval = inputs[((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3840)];
      } else {
        condval = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) * 2)] = condval;
      float condval_1;
      if (((1 < (((((int)blockIdx.x) / 48) * 6) + (rh_0 * 2))) && (1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5))))) {
        condval_1 = inputs[((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3839)];
      } else {
        condval_1 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 1)] = condval_1;
      float condval_2;
      if ((((1 <= ((((((int)blockIdx.x) / 48) * 6) + (rh_0 * 2)) + ((((int)threadIdx.x) + 192) >> 8))) && (1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7)))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7)) < 15))) {
        condval_2 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 192) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 6) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3840)];
      } else {
        condval_2 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 384)] = condval_2;
      float condval_3;
      if ((((1 <= ((((((int)blockIdx.x) / 48) * 6) + (rh_0 * 2)) + ((((int)threadIdx.x) + 192) >> 8))) && (1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7)))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7)) < 15))) {
        condval_3 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 192) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 6) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3839)];
      } else {
        condval_3 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 385)] = condval_3;
      float condval_4;
      if (((1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7)) < 15))) {
        condval_4 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 384) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 4) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3840)];
      } else {
        condval_4 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 768)] = condval_4;
      float condval_5;
      if (((1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7)) < 15))) {
        condval_5 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 384) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 4) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3839)];
      } else {
        condval_5 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 769)] = condval_5;
      float condval_6;
      if ((((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5)) < 13)) {
        condval_6 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 576) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3328)];
      } else {
        condval_6 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 1152)] = condval_6;
      float condval_7;
      if ((((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5)) < 13)) {
        condval_7 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 576) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3327)];
      } else {
        condval_7 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 1153)] = condval_7;
      float condval_8;
      if ((1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5)))) {
        condval_8 = inputs[((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 6912)];
      } else {
        condval_8 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 1536)] = condval_8;
      float condval_9;
      if ((1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5)))) {
        condval_9 = inputs[((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 6913)];
      } else {
        condval_9 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 1537)] = condval_9;
      float condval_10;
      if (((1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7)) < 15))) {
        condval_10 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 960) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 6) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3840)];
      } else {
        condval_10 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 1920)] = condval_10;
      float condval_11;
      if (((1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 6) & 7)) < 15))) {
        condval_11 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 960) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 6) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3839)];
      } else {
        condval_11 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 1921)] = condval_11;
      float condval_12;
      if (((((((((int)blockIdx.x) / 48) * 2) + (((rh_0 * 2) + ((((int)threadIdx.x) + 1152) >> 8)) / 3)) < 5) && (1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7)))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7)) < 15))) {
        condval_12 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 1152) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 4) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3840)];
      } else {
        condval_12 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 2304)] = condval_12;
      float condval_13;
      if (((((((((int)blockIdx.x) / 48) * 2) + (((rh_0 * 2) + ((((int)threadIdx.x) + 1152) >> 8)) / 3)) < 5) && (1 <= ((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7)))) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((((int)threadIdx.x) >> 5) + 4) & 7)) < 15))) {
        condval_13 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 1152) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((((int)threadIdx.x) >> 5) + 4) & 7) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3839)];
      } else {
        condval_13 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 2305)] = condval_13;
      float condval_14;
      if ((((((((int)blockIdx.x) / 48) * 2) + (((rh_0 * 2) + ((((int)threadIdx.x) + 1344) >> 8)) / 3)) < 5) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5)) < 13))) {
        condval_14 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 1344) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3328)];
      } else {
        condval_14 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 2688)] = condval_14;
      float condval_15;
      if ((((((((int)blockIdx.x) / 48) * 2) + (((rh_0 * 2) + ((((int)threadIdx.x) + 1344) >> 8)) / 3)) < 5) && (((((((int)blockIdx.x) % 48) >> 4) * 4) + (((int)threadIdx.x) >> 5)) < 13))) {
        condval_15 = inputs[(((((((((((int)blockIdx.x) / 48) * 21504) + (rh_0 * 7168)) + (((((int)threadIdx.x) + 1344) >> 8) * 3584)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_0 * 64)) + ((((int)threadIdx.x) & 31) * 2)) - 3327)];
      } else {
        condval_15 = 0.000000e+00f;
      }
      PadInput_shared[((((int)threadIdx.x) * 2) + 2689)] = condval_15;
      *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + (((((rh_0 * 196608) + (rc_0 * 16384)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(weight + ((((((rh_0 * 196608) + (((((int)threadIdx.x) + 192) >> 8) * 65536)) + (rc_0 * 16384)) + ((((((int)threadIdx.x) >> 2) + 48) & 63) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(weight + ((((((rh_0 * 196608) + (((((int)threadIdx.x) + 384) >> 8) * 65536)) + (rc_0 * 16384)) + ((((((int)threadIdx.x) >> 2) + 32) & 63) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(weight + (((((((rh_0 * 196608) + (((((int)threadIdx.x) + 576) >> 8) * 65536)) + (rc_0 * 16384)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 4096));
      __syncthreads();
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64))] * weight_shared[(((int)threadIdx.x) & 15)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 128)] * weight_shared[(((int)threadIdx.x) & 15)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 1)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 129)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 2)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 130)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 3)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 131)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 4)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 132)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 5)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 133)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 6)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 134)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 7)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 135)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 8)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 136)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 9)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 137)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 10)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 138)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 11)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 139)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 12)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 140)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 13)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 141)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 14)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 142)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 15)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 143)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 16)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 144)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 17)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 145)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 18)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 146)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 19)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 147)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 20)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 148)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 21)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 149)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 22)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 150)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 23)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 151)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 24)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 152)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 25)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 153)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 26)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 154)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 27)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 155)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 28)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 156)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 29)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 157)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 30)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 158)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 31)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 159)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 32)] * weight_shared[((((int)threadIdx.x) & 15) + 512)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 160)] * weight_shared[((((int)threadIdx.x) & 15) + 512)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 33)] * weight_shared[((((int)threadIdx.x) & 15) + 528)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 161)] * weight_shared[((((int)threadIdx.x) & 15) + 528)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 34)] * weight_shared[((((int)threadIdx.x) & 15) + 544)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 162)] * weight_shared[((((int)threadIdx.x) & 15) + 544)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 35)] * weight_shared[((((int)threadIdx.x) & 15) + 560)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 163)] * weight_shared[((((int)threadIdx.x) & 15) + 560)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 36)] * weight_shared[((((int)threadIdx.x) & 15) + 576)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 164)] * weight_shared[((((int)threadIdx.x) & 15) + 576)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 37)] * weight_shared[((((int)threadIdx.x) & 15) + 592)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 165)] * weight_shared[((((int)threadIdx.x) & 15) + 592)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 38)] * weight_shared[((((int)threadIdx.x) & 15) + 608)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 166)] * weight_shared[((((int)threadIdx.x) & 15) + 608)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 39)] * weight_shared[((((int)threadIdx.x) & 15) + 624)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 167)] * weight_shared[((((int)threadIdx.x) & 15) + 624)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 40)] * weight_shared[((((int)threadIdx.x) & 15) + 640)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 168)] * weight_shared[((((int)threadIdx.x) & 15) + 640)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 41)] * weight_shared[((((int)threadIdx.x) & 15) + 656)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 169)] * weight_shared[((((int)threadIdx.x) & 15) + 656)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 42)] * weight_shared[((((int)threadIdx.x) & 15) + 672)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 170)] * weight_shared[((((int)threadIdx.x) & 15) + 672)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 43)] * weight_shared[((((int)threadIdx.x) & 15) + 688)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 171)] * weight_shared[((((int)threadIdx.x) & 15) + 688)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 44)] * weight_shared[((((int)threadIdx.x) & 15) + 704)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 172)] * weight_shared[((((int)threadIdx.x) & 15) + 704)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 45)] * weight_shared[((((int)threadIdx.x) & 15) + 720)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 173)] * weight_shared[((((int)threadIdx.x) & 15) + 720)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 46)] * weight_shared[((((int)threadIdx.x) & 15) + 736)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 174)] * weight_shared[((((int)threadIdx.x) & 15) + 736)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 47)] * weight_shared[((((int)threadIdx.x) & 15) + 752)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 175)] * weight_shared[((((int)threadIdx.x) & 15) + 752)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 48)] * weight_shared[((((int)threadIdx.x) & 15) + 768)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 176)] * weight_shared[((((int)threadIdx.x) & 15) + 768)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 49)] * weight_shared[((((int)threadIdx.x) & 15) + 784)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 177)] * weight_shared[((((int)threadIdx.x) & 15) + 784)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 50)] * weight_shared[((((int)threadIdx.x) & 15) + 800)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 178)] * weight_shared[((((int)threadIdx.x) & 15) + 800)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 51)] * weight_shared[((((int)threadIdx.x) & 15) + 816)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 179)] * weight_shared[((((int)threadIdx.x) & 15) + 816)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 52)] * weight_shared[((((int)threadIdx.x) & 15) + 832)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 180)] * weight_shared[((((int)threadIdx.x) & 15) + 832)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 53)] * weight_shared[((((int)threadIdx.x) & 15) + 848)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 181)] * weight_shared[((((int)threadIdx.x) & 15) + 848)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 54)] * weight_shared[((((int)threadIdx.x) & 15) + 864)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 182)] * weight_shared[((((int)threadIdx.x) & 15) + 864)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 55)] * weight_shared[((((int)threadIdx.x) & 15) + 880)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 183)] * weight_shared[((((int)threadIdx.x) & 15) + 880)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 56)] * weight_shared[((((int)threadIdx.x) & 15) + 896)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 184)] * weight_shared[((((int)threadIdx.x) & 15) + 896)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 57)] * weight_shared[((((int)threadIdx.x) & 15) + 912)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 185)] * weight_shared[((((int)threadIdx.x) & 15) + 912)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 58)] * weight_shared[((((int)threadIdx.x) & 15) + 928)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 186)] * weight_shared[((((int)threadIdx.x) & 15) + 928)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 59)] * weight_shared[((((int)threadIdx.x) & 15) + 944)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 187)] * weight_shared[((((int)threadIdx.x) & 15) + 944)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 60)] * weight_shared[((((int)threadIdx.x) & 15) + 960)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 188)] * weight_shared[((((int)threadIdx.x) & 15) + 960)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 61)] * weight_shared[((((int)threadIdx.x) & 15) + 976)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 189)] * weight_shared[((((int)threadIdx.x) & 15) + 976)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 62)] * weight_shared[((((int)threadIdx.x) & 15) + 992)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 190)] * weight_shared[((((int)threadIdx.x) & 15) + 992)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 63)] * weight_shared[((((int)threadIdx.x) & 15) + 1008)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 191)] * weight_shared[((((int)threadIdx.x) & 15) + 1008)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 128)] * weight_shared[((((int)threadIdx.x) & 15) + 1024)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 256)] * weight_shared[((((int)threadIdx.x) & 15) + 1024)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 129)] * weight_shared[((((int)threadIdx.x) & 15) + 1040)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 257)] * weight_shared[((((int)threadIdx.x) & 15) + 1040)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 130)] * weight_shared[((((int)threadIdx.x) & 15) + 1056)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 258)] * weight_shared[((((int)threadIdx.x) & 15) + 1056)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 131)] * weight_shared[((((int)threadIdx.x) & 15) + 1072)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 259)] * weight_shared[((((int)threadIdx.x) & 15) + 1072)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 132)] * weight_shared[((((int)threadIdx.x) & 15) + 1088)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 260)] * weight_shared[((((int)threadIdx.x) & 15) + 1088)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 133)] * weight_shared[((((int)threadIdx.x) & 15) + 1104)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 261)] * weight_shared[((((int)threadIdx.x) & 15) + 1104)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 134)] * weight_shared[((((int)threadIdx.x) & 15) + 1120)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 262)] * weight_shared[((((int)threadIdx.x) & 15) + 1120)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 135)] * weight_shared[((((int)threadIdx.x) & 15) + 1136)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 263)] * weight_shared[((((int)threadIdx.x) & 15) + 1136)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 136)] * weight_shared[((((int)threadIdx.x) & 15) + 1152)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 264)] * weight_shared[((((int)threadIdx.x) & 15) + 1152)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 137)] * weight_shared[((((int)threadIdx.x) & 15) + 1168)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 265)] * weight_shared[((((int)threadIdx.x) & 15) + 1168)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 138)] * weight_shared[((((int)threadIdx.x) & 15) + 1184)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 266)] * weight_shared[((((int)threadIdx.x) & 15) + 1184)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 139)] * weight_shared[((((int)threadIdx.x) & 15) + 1200)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 267)] * weight_shared[((((int)threadIdx.x) & 15) + 1200)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 140)] * weight_shared[((((int)threadIdx.x) & 15) + 1216)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 268)] * weight_shared[((((int)threadIdx.x) & 15) + 1216)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 141)] * weight_shared[((((int)threadIdx.x) & 15) + 1232)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 269)] * weight_shared[((((int)threadIdx.x) & 15) + 1232)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 142)] * weight_shared[((((int)threadIdx.x) & 15) + 1248)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 270)] * weight_shared[((((int)threadIdx.x) & 15) + 1248)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 143)] * weight_shared[((((int)threadIdx.x) & 15) + 1264)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 271)] * weight_shared[((((int)threadIdx.x) & 15) + 1264)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 144)] * weight_shared[((((int)threadIdx.x) & 15) + 1280)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 272)] * weight_shared[((((int)threadIdx.x) & 15) + 1280)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 145)] * weight_shared[((((int)threadIdx.x) & 15) + 1296)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 273)] * weight_shared[((((int)threadIdx.x) & 15) + 1296)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 146)] * weight_shared[((((int)threadIdx.x) & 15) + 1312)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 274)] * weight_shared[((((int)threadIdx.x) & 15) + 1312)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 147)] * weight_shared[((((int)threadIdx.x) & 15) + 1328)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 275)] * weight_shared[((((int)threadIdx.x) & 15) + 1328)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 148)] * weight_shared[((((int)threadIdx.x) & 15) + 1344)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 276)] * weight_shared[((((int)threadIdx.x) & 15) + 1344)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 149)] * weight_shared[((((int)threadIdx.x) & 15) + 1360)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 277)] * weight_shared[((((int)threadIdx.x) & 15) + 1360)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 150)] * weight_shared[((((int)threadIdx.x) & 15) + 1376)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 278)] * weight_shared[((((int)threadIdx.x) & 15) + 1376)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 151)] * weight_shared[((((int)threadIdx.x) & 15) + 1392)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 279)] * weight_shared[((((int)threadIdx.x) & 15) + 1392)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 152)] * weight_shared[((((int)threadIdx.x) & 15) + 1408)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 280)] * weight_shared[((((int)threadIdx.x) & 15) + 1408)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 153)] * weight_shared[((((int)threadIdx.x) & 15) + 1424)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 281)] * weight_shared[((((int)threadIdx.x) & 15) + 1424)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 154)] * weight_shared[((((int)threadIdx.x) & 15) + 1440)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 282)] * weight_shared[((((int)threadIdx.x) & 15) + 1440)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 155)] * weight_shared[((((int)threadIdx.x) & 15) + 1456)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 283)] * weight_shared[((((int)threadIdx.x) & 15) + 1456)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 156)] * weight_shared[((((int)threadIdx.x) & 15) + 1472)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 284)] * weight_shared[((((int)threadIdx.x) & 15) + 1472)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 157)] * weight_shared[((((int)threadIdx.x) & 15) + 1488)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 285)] * weight_shared[((((int)threadIdx.x) & 15) + 1488)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 158)] * weight_shared[((((int)threadIdx.x) & 15) + 1504)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 286)] * weight_shared[((((int)threadIdx.x) & 15) + 1504)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 159)] * weight_shared[((((int)threadIdx.x) & 15) + 1520)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 287)] * weight_shared[((((int)threadIdx.x) & 15) + 1520)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 160)] * weight_shared[((((int)threadIdx.x) & 15) + 1536)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 288)] * weight_shared[((((int)threadIdx.x) & 15) + 1536)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 161)] * weight_shared[((((int)threadIdx.x) & 15) + 1552)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 289)] * weight_shared[((((int)threadIdx.x) & 15) + 1552)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 162)] * weight_shared[((((int)threadIdx.x) & 15) + 1568)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 290)] * weight_shared[((((int)threadIdx.x) & 15) + 1568)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 163)] * weight_shared[((((int)threadIdx.x) & 15) + 1584)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 291)] * weight_shared[((((int)threadIdx.x) & 15) + 1584)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 164)] * weight_shared[((((int)threadIdx.x) & 15) + 1600)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 292)] * weight_shared[((((int)threadIdx.x) & 15) + 1600)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 165)] * weight_shared[((((int)threadIdx.x) & 15) + 1616)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 293)] * weight_shared[((((int)threadIdx.x) & 15) + 1616)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 166)] * weight_shared[((((int)threadIdx.x) & 15) + 1632)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 294)] * weight_shared[((((int)threadIdx.x) & 15) + 1632)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 167)] * weight_shared[((((int)threadIdx.x) & 15) + 1648)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 295)] * weight_shared[((((int)threadIdx.x) & 15) + 1648)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 168)] * weight_shared[((((int)threadIdx.x) & 15) + 1664)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 296)] * weight_shared[((((int)threadIdx.x) & 15) + 1664)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 169)] * weight_shared[((((int)threadIdx.x) & 15) + 1680)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 297)] * weight_shared[((((int)threadIdx.x) & 15) + 1680)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 170)] * weight_shared[((((int)threadIdx.x) & 15) + 1696)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 298)] * weight_shared[((((int)threadIdx.x) & 15) + 1696)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 171)] * weight_shared[((((int)threadIdx.x) & 15) + 1712)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 299)] * weight_shared[((((int)threadIdx.x) & 15) + 1712)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 172)] * weight_shared[((((int)threadIdx.x) & 15) + 1728)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 300)] * weight_shared[((((int)threadIdx.x) & 15) + 1728)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 173)] * weight_shared[((((int)threadIdx.x) & 15) + 1744)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 301)] * weight_shared[((((int)threadIdx.x) & 15) + 1744)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 174)] * weight_shared[((((int)threadIdx.x) & 15) + 1760)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 302)] * weight_shared[((((int)threadIdx.x) & 15) + 1760)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 175)] * weight_shared[((((int)threadIdx.x) & 15) + 1776)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 303)] * weight_shared[((((int)threadIdx.x) & 15) + 1776)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 176)] * weight_shared[((((int)threadIdx.x) & 15) + 1792)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 304)] * weight_shared[((((int)threadIdx.x) & 15) + 1792)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 177)] * weight_shared[((((int)threadIdx.x) & 15) + 1808)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 305)] * weight_shared[((((int)threadIdx.x) & 15) + 1808)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 178)] * weight_shared[((((int)threadIdx.x) & 15) + 1824)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 306)] * weight_shared[((((int)threadIdx.x) & 15) + 1824)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 179)] * weight_shared[((((int)threadIdx.x) & 15) + 1840)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 307)] * weight_shared[((((int)threadIdx.x) & 15) + 1840)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 180)] * weight_shared[((((int)threadIdx.x) & 15) + 1856)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 308)] * weight_shared[((((int)threadIdx.x) & 15) + 1856)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 181)] * weight_shared[((((int)threadIdx.x) & 15) + 1872)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 309)] * weight_shared[((((int)threadIdx.x) & 15) + 1872)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 182)] * weight_shared[((((int)threadIdx.x) & 15) + 1888)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 310)] * weight_shared[((((int)threadIdx.x) & 15) + 1888)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 183)] * weight_shared[((((int)threadIdx.x) & 15) + 1904)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 311)] * weight_shared[((((int)threadIdx.x) & 15) + 1904)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 184)] * weight_shared[((((int)threadIdx.x) & 15) + 1920)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 312)] * weight_shared[((((int)threadIdx.x) & 15) + 1920)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 185)] * weight_shared[((((int)threadIdx.x) & 15) + 1936)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 313)] * weight_shared[((((int)threadIdx.x) & 15) + 1936)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 186)] * weight_shared[((((int)threadIdx.x) & 15) + 1952)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 314)] * weight_shared[((((int)threadIdx.x) & 15) + 1952)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 187)] * weight_shared[((((int)threadIdx.x) & 15) + 1968)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 315)] * weight_shared[((((int)threadIdx.x) & 15) + 1968)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 188)] * weight_shared[((((int)threadIdx.x) & 15) + 1984)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 316)] * weight_shared[((((int)threadIdx.x) & 15) + 1984)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 189)] * weight_shared[((((int)threadIdx.x) & 15) + 2000)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 317)] * weight_shared[((((int)threadIdx.x) & 15) + 2000)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 190)] * weight_shared[((((int)threadIdx.x) & 15) + 2016)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 318)] * weight_shared[((((int)threadIdx.x) & 15) + 2016)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 191)] * weight_shared[((((int)threadIdx.x) & 15) + 2032)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 319)] * weight_shared[((((int)threadIdx.x) & 15) + 2032)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 256)] * weight_shared[((((int)threadIdx.x) & 15) + 2048)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 384)] * weight_shared[((((int)threadIdx.x) & 15) + 2048)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 257)] * weight_shared[((((int)threadIdx.x) & 15) + 2064)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 385)] * weight_shared[((((int)threadIdx.x) & 15) + 2064)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 258)] * weight_shared[((((int)threadIdx.x) & 15) + 2080)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 386)] * weight_shared[((((int)threadIdx.x) & 15) + 2080)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 259)] * weight_shared[((((int)threadIdx.x) & 15) + 2096)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 387)] * weight_shared[((((int)threadIdx.x) & 15) + 2096)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 260)] * weight_shared[((((int)threadIdx.x) & 15) + 2112)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 388)] * weight_shared[((((int)threadIdx.x) & 15) + 2112)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 261)] * weight_shared[((((int)threadIdx.x) & 15) + 2128)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 389)] * weight_shared[((((int)threadIdx.x) & 15) + 2128)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 262)] * weight_shared[((((int)threadIdx.x) & 15) + 2144)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 390)] * weight_shared[((((int)threadIdx.x) & 15) + 2144)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 263)] * weight_shared[((((int)threadIdx.x) & 15) + 2160)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 391)] * weight_shared[((((int)threadIdx.x) & 15) + 2160)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 264)] * weight_shared[((((int)threadIdx.x) & 15) + 2176)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 392)] * weight_shared[((((int)threadIdx.x) & 15) + 2176)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 265)] * weight_shared[((((int)threadIdx.x) & 15) + 2192)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 393)] * weight_shared[((((int)threadIdx.x) & 15) + 2192)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 266)] * weight_shared[((((int)threadIdx.x) & 15) + 2208)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 394)] * weight_shared[((((int)threadIdx.x) & 15) + 2208)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 267)] * weight_shared[((((int)threadIdx.x) & 15) + 2224)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 395)] * weight_shared[((((int)threadIdx.x) & 15) + 2224)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 268)] * weight_shared[((((int)threadIdx.x) & 15) + 2240)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 396)] * weight_shared[((((int)threadIdx.x) & 15) + 2240)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 269)] * weight_shared[((((int)threadIdx.x) & 15) + 2256)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 397)] * weight_shared[((((int)threadIdx.x) & 15) + 2256)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 270)] * weight_shared[((((int)threadIdx.x) & 15) + 2272)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 398)] * weight_shared[((((int)threadIdx.x) & 15) + 2272)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 271)] * weight_shared[((((int)threadIdx.x) & 15) + 2288)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 399)] * weight_shared[((((int)threadIdx.x) & 15) + 2288)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 272)] * weight_shared[((((int)threadIdx.x) & 15) + 2304)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 400)] * weight_shared[((((int)threadIdx.x) & 15) + 2304)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 273)] * weight_shared[((((int)threadIdx.x) & 15) + 2320)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 401)] * weight_shared[((((int)threadIdx.x) & 15) + 2320)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 274)] * weight_shared[((((int)threadIdx.x) & 15) + 2336)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 402)] * weight_shared[((((int)threadIdx.x) & 15) + 2336)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 275)] * weight_shared[((((int)threadIdx.x) & 15) + 2352)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 403)] * weight_shared[((((int)threadIdx.x) & 15) + 2352)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 276)] * weight_shared[((((int)threadIdx.x) & 15) + 2368)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 404)] * weight_shared[((((int)threadIdx.x) & 15) + 2368)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 277)] * weight_shared[((((int)threadIdx.x) & 15) + 2384)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 405)] * weight_shared[((((int)threadIdx.x) & 15) + 2384)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 278)] * weight_shared[((((int)threadIdx.x) & 15) + 2400)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 406)] * weight_shared[((((int)threadIdx.x) & 15) + 2400)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 279)] * weight_shared[((((int)threadIdx.x) & 15) + 2416)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 407)] * weight_shared[((((int)threadIdx.x) & 15) + 2416)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 280)] * weight_shared[((((int)threadIdx.x) & 15) + 2432)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 408)] * weight_shared[((((int)threadIdx.x) & 15) + 2432)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 281)] * weight_shared[((((int)threadIdx.x) & 15) + 2448)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 409)] * weight_shared[((((int)threadIdx.x) & 15) + 2448)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 282)] * weight_shared[((((int)threadIdx.x) & 15) + 2464)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 410)] * weight_shared[((((int)threadIdx.x) & 15) + 2464)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 283)] * weight_shared[((((int)threadIdx.x) & 15) + 2480)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 411)] * weight_shared[((((int)threadIdx.x) & 15) + 2480)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 284)] * weight_shared[((((int)threadIdx.x) & 15) + 2496)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 412)] * weight_shared[((((int)threadIdx.x) & 15) + 2496)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 285)] * weight_shared[((((int)threadIdx.x) & 15) + 2512)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 413)] * weight_shared[((((int)threadIdx.x) & 15) + 2512)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 286)] * weight_shared[((((int)threadIdx.x) & 15) + 2528)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 414)] * weight_shared[((((int)threadIdx.x) & 15) + 2528)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 287)] * weight_shared[((((int)threadIdx.x) & 15) + 2544)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 415)] * weight_shared[((((int)threadIdx.x) & 15) + 2544)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 288)] * weight_shared[((((int)threadIdx.x) & 15) + 2560)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 416)] * weight_shared[((((int)threadIdx.x) & 15) + 2560)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 289)] * weight_shared[((((int)threadIdx.x) & 15) + 2576)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 417)] * weight_shared[((((int)threadIdx.x) & 15) + 2576)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 290)] * weight_shared[((((int)threadIdx.x) & 15) + 2592)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 418)] * weight_shared[((((int)threadIdx.x) & 15) + 2592)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 291)] * weight_shared[((((int)threadIdx.x) & 15) + 2608)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 419)] * weight_shared[((((int)threadIdx.x) & 15) + 2608)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 292)] * weight_shared[((((int)threadIdx.x) & 15) + 2624)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 420)] * weight_shared[((((int)threadIdx.x) & 15) + 2624)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 293)] * weight_shared[((((int)threadIdx.x) & 15) + 2640)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 421)] * weight_shared[((((int)threadIdx.x) & 15) + 2640)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 294)] * weight_shared[((((int)threadIdx.x) & 15) + 2656)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 422)] * weight_shared[((((int)threadIdx.x) & 15) + 2656)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 295)] * weight_shared[((((int)threadIdx.x) & 15) + 2672)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 423)] * weight_shared[((((int)threadIdx.x) & 15) + 2672)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 296)] * weight_shared[((((int)threadIdx.x) & 15) + 2688)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 424)] * weight_shared[((((int)threadIdx.x) & 15) + 2688)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 297)] * weight_shared[((((int)threadIdx.x) & 15) + 2704)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 425)] * weight_shared[((((int)threadIdx.x) & 15) + 2704)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 298)] * weight_shared[((((int)threadIdx.x) & 15) + 2720)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 426)] * weight_shared[((((int)threadIdx.x) & 15) + 2720)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 299)] * weight_shared[((((int)threadIdx.x) & 15) + 2736)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 427)] * weight_shared[((((int)threadIdx.x) & 15) + 2736)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 300)] * weight_shared[((((int)threadIdx.x) & 15) + 2752)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 428)] * weight_shared[((((int)threadIdx.x) & 15) + 2752)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 301)] * weight_shared[((((int)threadIdx.x) & 15) + 2768)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 429)] * weight_shared[((((int)threadIdx.x) & 15) + 2768)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 302)] * weight_shared[((((int)threadIdx.x) & 15) + 2784)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 430)] * weight_shared[((((int)threadIdx.x) & 15) + 2784)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 303)] * weight_shared[((((int)threadIdx.x) & 15) + 2800)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 431)] * weight_shared[((((int)threadIdx.x) & 15) + 2800)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 304)] * weight_shared[((((int)threadIdx.x) & 15) + 2816)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 432)] * weight_shared[((((int)threadIdx.x) & 15) + 2816)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 305)] * weight_shared[((((int)threadIdx.x) & 15) + 2832)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 433)] * weight_shared[((((int)threadIdx.x) & 15) + 2832)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 306)] * weight_shared[((((int)threadIdx.x) & 15) + 2848)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 434)] * weight_shared[((((int)threadIdx.x) & 15) + 2848)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 307)] * weight_shared[((((int)threadIdx.x) & 15) + 2864)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 435)] * weight_shared[((((int)threadIdx.x) & 15) + 2864)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 308)] * weight_shared[((((int)threadIdx.x) & 15) + 2880)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 436)] * weight_shared[((((int)threadIdx.x) & 15) + 2880)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 309)] * weight_shared[((((int)threadIdx.x) & 15) + 2896)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 437)] * weight_shared[((((int)threadIdx.x) & 15) + 2896)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 310)] * weight_shared[((((int)threadIdx.x) & 15) + 2912)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 438)] * weight_shared[((((int)threadIdx.x) & 15) + 2912)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 311)] * weight_shared[((((int)threadIdx.x) & 15) + 2928)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 439)] * weight_shared[((((int)threadIdx.x) & 15) + 2928)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 312)] * weight_shared[((((int)threadIdx.x) & 15) + 2944)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 440)] * weight_shared[((((int)threadIdx.x) & 15) + 2944)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 313)] * weight_shared[((((int)threadIdx.x) & 15) + 2960)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 441)] * weight_shared[((((int)threadIdx.x) & 15) + 2960)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 314)] * weight_shared[((((int)threadIdx.x) & 15) + 2976)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 442)] * weight_shared[((((int)threadIdx.x) & 15) + 2976)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 315)] * weight_shared[((((int)threadIdx.x) & 15) + 2992)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 443)] * weight_shared[((((int)threadIdx.x) & 15) + 2992)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 316)] * weight_shared[((((int)threadIdx.x) & 15) + 3008)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 444)] * weight_shared[((((int)threadIdx.x) & 15) + 3008)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 317)] * weight_shared[((((int)threadIdx.x) & 15) + 3024)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 445)] * weight_shared[((((int)threadIdx.x) & 15) + 3024)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 318)] * weight_shared[((((int)threadIdx.x) & 15) + 3040)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 446)] * weight_shared[((((int)threadIdx.x) & 15) + 3040)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 319)] * weight_shared[((((int)threadIdx.x) & 15) + 3056)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 447)] * weight_shared[((((int)threadIdx.x) & 15) + 3056)]));
    }
  }
  conv2d_nhwc[(((((((((int)blockIdx.x) / 48) * 18432) + ((((int)threadIdx.x) >> 5) * 3072)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15))] = conv2d_nhwc_local[0];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 48) * 18432) + ((((int)threadIdx.x) >> 5) * 3072)) + (((((int)blockIdx.x) % 48) >> 4) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 512)] = conv2d_nhwc_local[1];
}

