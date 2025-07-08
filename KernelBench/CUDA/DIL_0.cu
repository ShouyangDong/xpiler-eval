
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
extern "C" __global__ void __launch_bounds__(218) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(218) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[4];
  __shared__ float PadInput_shared[2748];
  __shared__ float weight_shared[672];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  float condval;
  if (((15 < ((int)blockIdx.x)) && (9 <= ((int)threadIdx.x)))) {
    condval = inputs[((((((int)blockIdx.x) >> 3) * 1344) + ((int)threadIdx.x)) - 2025)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  float condval_1;
  if ((15 < ((int)blockIdx.x))) {
    condval_1 = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (((((int)threadIdx.x) + 218) / 3) * 3)) + ((((int)threadIdx.x) + 2) % 3)) - 2025)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[((((((int)threadIdx.x) + 218) / 3) * 3) + ((((int)threadIdx.x) + 2) % 3))] = condval_1;
  float condval_2;
  if ((15 < ((int)blockIdx.x))) {
    condval_2 = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (((((int)threadIdx.x) + 436) / 3) * 3)) + ((((int)threadIdx.x) + 1) % 3)) - 2025)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[((((((int)threadIdx.x) + 436) / 3) * 3) + ((((int)threadIdx.x) + 1) % 3))] = condval_2;
  if (((int)threadIdx.x) < 33) {
    float condval_3;
    if (((15 < ((int)blockIdx.x)) && (((int)threadIdx.x) < 27))) {
      condval_3 = inputs[((((((int)blockIdx.x) >> 3) * 1344) + ((int)threadIdx.x)) - 1371)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 654)] = condval_3;
  }
  if (((int)threadIdx.x) < 42) {
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((int)threadIdx.x) >> 1) * 64) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 4)));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float condval_4;
  if (((7 < ((int)blockIdx.x)) && (9 <= ((int)threadIdx.x)))) {
    condval_4 = inputs[((((((int)blockIdx.x) >> 3) * 1344) + ((int)threadIdx.x)) - 681)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 687)] = condval_4;
  float condval_5;
  if ((7 < ((int)blockIdx.x))) {
    condval_5 = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (((((int)threadIdx.x) + 218) / 3) * 3)) + ((((int)threadIdx.x) + 2) % 3)) - 681)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((((((int)threadIdx.x) + 218) / 3) * 3) + ((((int)threadIdx.x) + 2) % 3)) + 687)] = condval_5;
  float condval_6;
  if ((7 < ((int)blockIdx.x))) {
    condval_6 = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (((((int)threadIdx.x) + 436) / 3) * 3)) + ((((int)threadIdx.x) + 1) % 3)) - 681)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((((((int)threadIdx.x) + 436) / 3) * 3) + ((((int)threadIdx.x) + 1) % 3)) + 687)] = condval_6;
  if (((int)threadIdx.x) < 33) {
    float condval_7;
    if (((7 < ((int)blockIdx.x)) && (((int)threadIdx.x) < 27))) {
      condval_7 = inputs[((((((int)blockIdx.x) >> 3) * 1344) + ((int)threadIdx.x)) - 27)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1341)] = condval_7;
  }
  if (((int)threadIdx.x) < 42) {
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 168)) = *(float4*)(weight + (((((((int)threadIdx.x) >> 1) * 64) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 1344));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float condval_8;
  if ((9 <= ((int)threadIdx.x))) {
    condval_8 = inputs[((((((int)blockIdx.x) >> 3) * 1344) + ((int)threadIdx.x)) + 663)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 1374)] = condval_8;
  PadInput_shared[(((((((int)threadIdx.x) + 218) / 3) * 3) + ((((int)threadIdx.x) + 2) % 3)) + 1374)] = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (((((int)threadIdx.x) + 218) / 3) * 3)) + ((((int)threadIdx.x) + 2) % 3)) + 663)];
  PadInput_shared[(((((((int)threadIdx.x) + 436) / 3) * 3) + ((((int)threadIdx.x) + 1) % 3)) + 1374)] = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (((((int)threadIdx.x) + 436) / 3) * 3)) + ((((int)threadIdx.x) + 1) % 3)) + 663)];
  if (((int)threadIdx.x) < 33) {
    float condval_9;
    if ((((int)threadIdx.x) < 27)) {
      condval_9 = inputs[((((((int)blockIdx.x) >> 3) * 1344) + ((int)threadIdx.x)) + 1317)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 2028)] = condval_9;
  }
  if (((int)threadIdx.x) < 42) {
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 336)) = *(float4*)(weight + (((((((int)threadIdx.x) >> 1) * 64) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 2688));
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int rh_0_rw_0_rc_0_fused = 0; rh_0_rw_0_rc_0_fused < 4; ++rh_0_rw_0_rc_0_fused) {
    __syncthreads();
    float condval_10;
    if (((((((int)blockIdx.x) >> 3) + rh_0_rw_0_rc_0_fused) < 111) && (9 <= ((int)threadIdx.x)))) {
      condval_10 = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (rh_0_rw_0_rc_0_fused * 1344)) + ((int)threadIdx.x)) + 2007)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PadInput_shared[((((rh_0_rw_0_rc_0_fused + 3) & 3) * 687) + ((int)threadIdx.x))] = condval_10;
    float condval_11;
    if ((((((int)blockIdx.x) >> 3) + rh_0_rw_0_rc_0_fused) < 111)) {
      condval_11 = inputs[((((((((int)blockIdx.x) >> 3) * 1344) + (rh_0_rw_0_rc_0_fused * 1344)) + (((((int)threadIdx.x) + 218) / 3) * 3)) + ((((int)threadIdx.x) + 2) % 3)) + 2007)];
    } else {
      condval_11 = 0.000000e+00f;
    }
    PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 687) + (((((int)threadIdx.x) + 218) / 3) * 3)) + ((((int)threadIdx.x) + 2) % 3))] = condval_11;
    float condval_12;
    if ((((((int)blockIdx.x) >> 3) + rh_0_rw_0_rc_0_fused) < 111)) {
      condval_12 = inputs[((((((((int)blockIdx.x) >> 3) * 1344) + (rh_0_rw_0_rc_0_fused * 1344)) + (((((int)threadIdx.x) + 436) / 3) * 3)) + ((((int)threadIdx.x) + 1) % 3)) + 2007)];
    } else {
      condval_12 = 0.000000e+00f;
    }
    PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 687) + (((((int)threadIdx.x) + 436) / 3) * 3)) + ((((int)threadIdx.x) + 1) % 3))] = condval_12;
    if (((int)threadIdx.x) < 33) {
      float condval_13;
      if (((((((int)blockIdx.x) >> 3) + rh_0_rw_0_rc_0_fused) < 111) && (((int)threadIdx.x) < 27))) {
        condval_13 = inputs[(((((((int)blockIdx.x) >> 3) * 1344) + (rh_0_rw_0_rc_0_fused * 1344)) + ((int)threadIdx.x)) + 2661)];
      } else {
        condval_13 = 0.000000e+00f;
      }
      PadInput_shared[(((((rh_0_rw_0_rc_0_fused + 3) & 3) * 687) + ((int)threadIdx.x)) + 654)] = condval_13;
    }
    if (((int)threadIdx.x) < 42) {
      *(float4*)(weight_shared + ((((rh_0_rw_0_rc_0_fused + 3) & 3) * 168) + (((int)threadIdx.x) * 4))) = *(float4*)(weight + (((((rh_0_rw_0_rc_0_fused * 1344) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 4032));
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

    __syncthreads();
    for (int rc_1 = 0; rc_1 < 3; ++rc_1) {
      for (int co_3 = 0; co_3 < 2; ++co_3) {
        conv2d_nhwc_local[co_3] = (conv2d_nhwc_local[co_3] + (PadInput_shared[(((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1)] * weight_shared[((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3)]));
        conv2d_nhwc_local[(co_3 + 2)] = (conv2d_nhwc_local[(co_3 + 2)] + (PadInput_shared[(((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 4)]));
        conv2d_nhwc_local[co_3] = (conv2d_nhwc_local[co_3] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 6)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 24)]));
        conv2d_nhwc_local[(co_3 + 2)] = (conv2d_nhwc_local[(co_3 + 2)] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 6)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 28)]));
        conv2d_nhwc_local[co_3] = (conv2d_nhwc_local[co_3] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 12)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 48)]));
        conv2d_nhwc_local[(co_3 + 2)] = (conv2d_nhwc_local[(co_3 + 2)] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 12)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 52)]));
        conv2d_nhwc_local[co_3] = (conv2d_nhwc_local[co_3] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 18)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 72)]));
        conv2d_nhwc_local[(co_3 + 2)] = (conv2d_nhwc_local[(co_3 + 2)] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 18)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 76)]));
        conv2d_nhwc_local[co_3] = (conv2d_nhwc_local[co_3] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 24)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 96)]));
        conv2d_nhwc_local[(co_3 + 2)] = (conv2d_nhwc_local[(co_3 + 2)] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 24)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 100)]));
        conv2d_nhwc_local[co_3] = (conv2d_nhwc_local[co_3] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 30)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 120)]));
        conv2d_nhwc_local[(co_3 + 2)] = (conv2d_nhwc_local[(co_3 + 2)] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 30)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 124)]));
        conv2d_nhwc_local[co_3] = (conv2d_nhwc_local[co_3] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 36)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 144)]));
        conv2d_nhwc_local[(co_3 + 2)] = (conv2d_nhwc_local[(co_3 + 2)] + (PadInput_shared[((((rh_0_rw_0_rc_0_fused * 687) + ((((int)threadIdx.x) >> 1) * 6)) + rc_1) + 36)] * weight_shared[(((((rh_0_rw_0_rc_0_fused * 168) + (rc_1 * 8)) + ((((int)threadIdx.x) & 1) * 2)) + co_3) + 148)]));
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int rc_1_1 = 0; rc_1_1 < 3; ++rc_1_1) {
    for (int co_3_1 = 0; co_3_1 < 2; ++co_3_1) {
      conv2d_nhwc_local[co_3_1] = (conv2d_nhwc_local[co_3_1] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + rc_1_1)] * weight_shared[(((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1)]));
      conv2d_nhwc_local[(co_3_1 + 2)] = (conv2d_nhwc_local[(co_3_1 + 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + rc_1_1)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 4)]));
      conv2d_nhwc_local[co_3_1] = (conv2d_nhwc_local[co_3_1] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 6)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 24)]));
      conv2d_nhwc_local[(co_3_1 + 2)] = (conv2d_nhwc_local[(co_3_1 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 6)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 28)]));
      conv2d_nhwc_local[co_3_1] = (conv2d_nhwc_local[co_3_1] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 12)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 48)]));
      conv2d_nhwc_local[(co_3_1 + 2)] = (conv2d_nhwc_local[(co_3_1 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 12)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 52)]));
      conv2d_nhwc_local[co_3_1] = (conv2d_nhwc_local[co_3_1] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 18)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 72)]));
      conv2d_nhwc_local[(co_3_1 + 2)] = (conv2d_nhwc_local[(co_3_1 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 18)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 76)]));
      conv2d_nhwc_local[co_3_1] = (conv2d_nhwc_local[co_3_1] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 24)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 96)]));
      conv2d_nhwc_local[(co_3_1 + 2)] = (conv2d_nhwc_local[(co_3_1 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 24)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 100)]));
      conv2d_nhwc_local[co_3_1] = (conv2d_nhwc_local[co_3_1] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 30)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 120)]));
      conv2d_nhwc_local[(co_3_1 + 2)] = (conv2d_nhwc_local[(co_3_1 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 30)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 124)]));
      conv2d_nhwc_local[co_3_1] = (conv2d_nhwc_local[co_3_1] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 36)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 144)]));
      conv2d_nhwc_local[(co_3_1 + 2)] = (conv2d_nhwc_local[(co_3_1 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_1) + 36)] * weight_shared[((((rc_1_1 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_1) + 148)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int rc_1_2 = 0; rc_1_2 < 3; ++rc_1_2) {
    for (int co_3_2 = 0; co_3_2 < 2; ++co_3_2) {
      conv2d_nhwc_local[co_3_2] = (conv2d_nhwc_local[co_3_2] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 687)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 168)]));
      conv2d_nhwc_local[(co_3_2 + 2)] = (conv2d_nhwc_local[(co_3_2 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 687)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 172)]));
      conv2d_nhwc_local[co_3_2] = (conv2d_nhwc_local[co_3_2] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 693)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 192)]));
      conv2d_nhwc_local[(co_3_2 + 2)] = (conv2d_nhwc_local[(co_3_2 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 693)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 196)]));
      conv2d_nhwc_local[co_3_2] = (conv2d_nhwc_local[co_3_2] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 699)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 216)]));
      conv2d_nhwc_local[(co_3_2 + 2)] = (conv2d_nhwc_local[(co_3_2 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 699)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 220)]));
      conv2d_nhwc_local[co_3_2] = (conv2d_nhwc_local[co_3_2] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 705)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 240)]));
      conv2d_nhwc_local[(co_3_2 + 2)] = (conv2d_nhwc_local[(co_3_2 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 705)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 244)]));
      conv2d_nhwc_local[co_3_2] = (conv2d_nhwc_local[co_3_2] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 711)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 264)]));
      conv2d_nhwc_local[(co_3_2 + 2)] = (conv2d_nhwc_local[(co_3_2 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 711)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 268)]));
      conv2d_nhwc_local[co_3_2] = (conv2d_nhwc_local[co_3_2] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 717)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 288)]));
      conv2d_nhwc_local[(co_3_2 + 2)] = (conv2d_nhwc_local[(co_3_2 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 717)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 292)]));
      conv2d_nhwc_local[co_3_2] = (conv2d_nhwc_local[co_3_2] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 723)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 312)]));
      conv2d_nhwc_local[(co_3_2 + 2)] = (conv2d_nhwc_local[(co_3_2 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_2) + 723)] * weight_shared[((((rc_1_2 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_2) + 316)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int rc_1_3 = 0; rc_1_3 < 3; ++rc_1_3) {
    for (int co_3_3 = 0; co_3_3 < 2; ++co_3_3) {
      conv2d_nhwc_local[co_3_3] = (conv2d_nhwc_local[co_3_3] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1374)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 336)]));
      conv2d_nhwc_local[(co_3_3 + 2)] = (conv2d_nhwc_local[(co_3_3 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1374)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 340)]));
      conv2d_nhwc_local[co_3_3] = (conv2d_nhwc_local[co_3_3] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1380)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 360)]));
      conv2d_nhwc_local[(co_3_3 + 2)] = (conv2d_nhwc_local[(co_3_3 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1380)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 364)]));
      conv2d_nhwc_local[co_3_3] = (conv2d_nhwc_local[co_3_3] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1386)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 384)]));
      conv2d_nhwc_local[(co_3_3 + 2)] = (conv2d_nhwc_local[(co_3_3 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1386)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 388)]));
      conv2d_nhwc_local[co_3_3] = (conv2d_nhwc_local[co_3_3] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1392)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 408)]));
      conv2d_nhwc_local[(co_3_3 + 2)] = (conv2d_nhwc_local[(co_3_3 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1392)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 412)]));
      conv2d_nhwc_local[co_3_3] = (conv2d_nhwc_local[co_3_3] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1398)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 432)]));
      conv2d_nhwc_local[(co_3_3 + 2)] = (conv2d_nhwc_local[(co_3_3 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1398)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 436)]));
      conv2d_nhwc_local[co_3_3] = (conv2d_nhwc_local[co_3_3] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1404)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 456)]));
      conv2d_nhwc_local[(co_3_3 + 2)] = (conv2d_nhwc_local[(co_3_3 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1404)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 460)]));
      conv2d_nhwc_local[co_3_3] = (conv2d_nhwc_local[co_3_3] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1410)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 480)]));
      conv2d_nhwc_local[(co_3_3 + 2)] = (conv2d_nhwc_local[(co_3_3 + 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 1) * 6) + rc_1_3) + 1410)] * weight_shared[((((rc_1_3 * 8) + ((((int)threadIdx.x) & 1) * 2)) + co_3_3) + 484)]));
    }
  }
  conv2d_nhwc[(((((((int)blockIdx.x) >> 3) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 2))] = conv2d_nhwc_local[0];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 3) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + 4)] = conv2d_nhwc_local[2];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 3) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + 1)] = conv2d_nhwc_local[1];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 3) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + 5)] = conv2d_nhwc_local[3];
}

