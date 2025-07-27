
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
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ conv2d_capsule_nhwijc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ conv2d_capsule_nhwijc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_capsule_nhwijc_local[16];
  __shared__ float PadInput_shared[5440];
  __shared__ float weight_shared[2304];
  conv2d_capsule_nhwijc_local[0] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[2] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[4] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[6] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[8] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[10] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[12] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[14] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[1] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[3] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[5] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[7] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[9] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[11] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[13] = 0.000000e+00f;
  conv2d_capsule_nhwijc_local[15] = 0.000000e+00f;
  for (int cap_k_0 = 0; cap_k_0 < 4; ++cap_k_0) {
    for (int rc_0 = 0; rc_0 < 4; ++rc_0) {
      __syncthreads();
      float condval;
      if (((4 <= (((int)blockIdx.x) & 15)) && (32 <= ((int)threadIdx.x)))) {
        condval = inputs[((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval = 0.000000e+00f;
      }
      PadInput_shared[((int)threadIdx.x)] = condval;
      float condval_1;
      if ((4 <= (((int)blockIdx.x) & 15))) {
        condval_1 = inputs[((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4608)];
      } else {
        condval_1 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 256)] = condval_1;
      float condval_2;
      if (((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 4) + ((((int)threadIdx.x) + 512) / 544))) && (1 <= (((((int)threadIdx.x) >> 5) + 16) % 17)))) {
        condval_2 = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 512) / 544) * 8192)) + ((((((int)threadIdx.x) >> 5) + 16) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_2 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 512)] = condval_2;
      PadInput_shared[(((int)threadIdx.x) + 768)] = inputs[(((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 768) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 5120)];
      float condval_3;
      if ((1 <= (((((int)threadIdx.x) >> 5) + 15) % 17))) {
        condval_3 = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 1024) / 544) * 8192)) + ((((((int)threadIdx.x) >> 5) + 15) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_3 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 1024)] = condval_3;
      PadInput_shared[(((int)threadIdx.x) + 1280)] = inputs[(((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 1280) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 5632)];
      float condval_4;
      if ((1 <= (((((int)threadIdx.x) >> 5) + 14) % 17))) {
        condval_4 = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 1536) / 544) * 8192)) + ((((((int)threadIdx.x) >> 5) + 14) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_4 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 1536)] = condval_4;
      PadInput_shared[(((int)threadIdx.x) + 1792)] = inputs[(((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 1792) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 6144)];
      float condval_5;
      if ((1 <= (((((int)threadIdx.x) >> 5) + 13) % 17))) {
        condval_5 = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 2048) / 544) * 8192)) + ((((((int)threadIdx.x) >> 5) + 13) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_5 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 2048)] = condval_5;
      PadInput_shared[(((int)threadIdx.x) + 2304)] = inputs[(((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 2304) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 6656)];
      float condval_6;
      if (((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 4) + ((((((int)threadIdx.x) >> 5) + 80) % 85) / 17))) && (1 <= (((((int)threadIdx.x) >> 5) + 12) % 17)))) {
        condval_6 = inputs[(((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 2560) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((((int)threadIdx.x) >> 5) + 80) % 85) / 17) * 8192)) + ((((((int)threadIdx.x) >> 5) + 12) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_6 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 2560)] = condval_6;
      float condval_7;
      if ((3 < (((int)blockIdx.x) & 15))) {
        condval_7 = inputs[(((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 2816) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 7168)];
      } else {
        condval_7 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 2816)] = condval_7;
      float condval_8;
      if (((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 4) + ((((int)threadIdx.x) + 352) / 544))) && (1 <= (((((int)threadIdx.x) >> 5) + 11) % 17)))) {
        condval_8 = inputs[(((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 3072) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 352) / 544) * 8192)) + ((((((int)threadIdx.x) >> 5) + 11) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_8 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 3072)] = condval_8;
      PadInput_shared[(((int)threadIdx.x) + 3328)] = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 3328) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 608) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 7680)];
      float condval_9;
      if ((1 <= (((((int)threadIdx.x) >> 5) + 10) % 17))) {
        condval_9 = inputs[(((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 3584) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 864) / 544) * 8192)) + ((((((int)threadIdx.x) >> 5) + 10) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_9 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 3584)] = condval_9;
      PadInput_shared[(((int)threadIdx.x) + 3840)] = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 3840) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 1120) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8192)];
      PadInput_shared[(((int)threadIdx.x) + 4096)] = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 4096) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 1376) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4096)];
      float condval_10;
      if ((32 <= ((int)threadIdx.x))) {
        condval_10 = inputs[(((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 4352) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 15872)];
      } else {
        condval_10 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 4352)] = condval_10;
      PadInput_shared[(((int)threadIdx.x) + 4608)] = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 4608) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 1888) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 4608)];
      float condval_11;
      if ((1 <= (((((int)threadIdx.x) >> 5) + 16) % 17))) {
        condval_11 = inputs[(((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 4864) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 2144) / 544) * 8192)) + ((((((int)threadIdx.x) >> 5) + 16) % 17) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 8704)];
      } else {
        condval_11 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 4864)] = condval_11;
      PadInput_shared[(((int)threadIdx.x) + 5120)] = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 5120) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 2400) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 5120)];
      if (((int)threadIdx.x) < 64) {
        PadInput_shared[(((int)threadIdx.x) + 5376)] = inputs[((((((((((((int)blockIdx.x) >> 4) * 262144) + (((((int)threadIdx.x) + 5376) / 2720) * 131072)) + (((((int)blockIdx.x) & 15) >> 2) * 32768)) + (((((int)threadIdx.x) + 2656) / 544) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + (cap_k_0 * 32)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 1024)];
      }
      *(float2*)(weight_shared + (((int)threadIdx.x) * 2)) = *(float2*)(weight + (((((((((((int)threadIdx.x) >> 7) * 16384) + (cap_k_0 * 4096)) + (((((int)blockIdx.x) & 3) >> 1) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (rc_0 * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
      *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(weight + ((((((((((((int)threadIdx.x) >> 7) * 16384) + (cap_k_0 * 4096)) + (((((int)blockIdx.x) & 3) >> 1) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (rc_0 * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 32768));
      *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(float2*)(weight + ((((((((((((int)threadIdx.x) >> 7) * 16384) + (cap_k_0 * 4096)) + (((((int)blockIdx.x) & 3) >> 1) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (rc_0 * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 65536));
      *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1536)) = *(float2*)(weight + ((((((((((((int)threadIdx.x) >> 7) * 16384) + (cap_k_0 * 4096)) + (((((int)blockIdx.x) & 3) >> 1) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (rc_0 * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 98304));
      if (((int)threadIdx.x) < 128) {
        *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2048)) = *(float2*)(weight + ((((((((cap_k_0 * 4096) + (((((int)blockIdx.x) & 3) >> 1) * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_0 * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 131072));
      }
      __syncthreads();
      for (int rw_1 = 0; rw_1 < 3; ++rw_1) {
        for (int rc_1 = 0; rc_1 < 2; ++rc_1) {
          for (int n_3 = 0; n_3 < 2; ++n_3) {
            for (int rh_2 = 0; rh_2 < 3; ++rh_2) {
              conv2d_capsule_nhwijc_local[n_3] = (conv2d_capsule_nhwijc_local[n_3] + (PadInput_shared[(((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4))] * weight_shared[((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7))]));
              conv2d_capsule_nhwijc_local[(n_3 + 2)] = (conv2d_capsule_nhwijc_local[(n_3 + 2)] + (PadInput_shared[(((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4))] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 8)]));
              conv2d_capsule_nhwijc_local[(n_3 + 4)] = (conv2d_capsule_nhwijc_local[(n_3 + 4)] + (PadInput_shared[(((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4))] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 128)]));
              conv2d_capsule_nhwijc_local[(n_3 + 6)] = (conv2d_capsule_nhwijc_local[(n_3 + 6)] + (PadInput_shared[(((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4))] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 136)]));
              conv2d_capsule_nhwijc_local[(n_3 + 8)] = (conv2d_capsule_nhwijc_local[(n_3 + 8)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 256)] * weight_shared[((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7))]));
              conv2d_capsule_nhwijc_local[(n_3 + 10)] = (conv2d_capsule_nhwijc_local[(n_3 + 10)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 256)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 8)]));
              conv2d_capsule_nhwijc_local[(n_3 + 12)] = (conv2d_capsule_nhwijc_local[(n_3 + 12)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 256)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 128)]));
              conv2d_capsule_nhwijc_local[(n_3 + 14)] = (conv2d_capsule_nhwijc_local[(n_3 + 14)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 256)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 136)]));
              conv2d_capsule_nhwijc_local[n_3] = (conv2d_capsule_nhwijc_local[n_3] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 1)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 16)]));
              conv2d_capsule_nhwijc_local[(n_3 + 2)] = (conv2d_capsule_nhwijc_local[(n_3 + 2)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 1)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 24)]));
              conv2d_capsule_nhwijc_local[(n_3 + 4)] = (conv2d_capsule_nhwijc_local[(n_3 + 4)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 1)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 144)]));
              conv2d_capsule_nhwijc_local[(n_3 + 6)] = (conv2d_capsule_nhwijc_local[(n_3 + 6)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 1)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 152)]));
              conv2d_capsule_nhwijc_local[(n_3 + 8)] = (conv2d_capsule_nhwijc_local[(n_3 + 8)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 257)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 16)]));
              conv2d_capsule_nhwijc_local[(n_3 + 10)] = (conv2d_capsule_nhwijc_local[(n_3 + 10)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 257)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 24)]));
              conv2d_capsule_nhwijc_local[(n_3 + 12)] = (conv2d_capsule_nhwijc_local[(n_3 + 12)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 257)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 144)]));
              conv2d_capsule_nhwijc_local[(n_3 + 14)] = (conv2d_capsule_nhwijc_local[(n_3 + 14)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 257)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 152)]));
              conv2d_capsule_nhwijc_local[n_3] = (conv2d_capsule_nhwijc_local[n_3] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 2)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 32)]));
              conv2d_capsule_nhwijc_local[(n_3 + 2)] = (conv2d_capsule_nhwijc_local[(n_3 + 2)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 2)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 40)]));
              conv2d_capsule_nhwijc_local[(n_3 + 4)] = (conv2d_capsule_nhwijc_local[(n_3 + 4)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 2)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 160)]));
              conv2d_capsule_nhwijc_local[(n_3 + 6)] = (conv2d_capsule_nhwijc_local[(n_3 + 6)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 2)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 168)]));
              conv2d_capsule_nhwijc_local[(n_3 + 8)] = (conv2d_capsule_nhwijc_local[(n_3 + 8)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 258)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 32)]));
              conv2d_capsule_nhwijc_local[(n_3 + 10)] = (conv2d_capsule_nhwijc_local[(n_3 + 10)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 258)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 40)]));
              conv2d_capsule_nhwijc_local[(n_3 + 12)] = (conv2d_capsule_nhwijc_local[(n_3 + 12)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 258)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 160)]));
              conv2d_capsule_nhwijc_local[(n_3 + 14)] = (conv2d_capsule_nhwijc_local[(n_3 + 14)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 258)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 168)]));
              conv2d_capsule_nhwijc_local[n_3] = (conv2d_capsule_nhwijc_local[n_3] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 3)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 48)]));
              conv2d_capsule_nhwijc_local[(n_3 + 2)] = (conv2d_capsule_nhwijc_local[(n_3 + 2)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 3)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 56)]));
              conv2d_capsule_nhwijc_local[(n_3 + 4)] = (conv2d_capsule_nhwijc_local[(n_3 + 4)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 3)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 176)]));
              conv2d_capsule_nhwijc_local[(n_3 + 6)] = (conv2d_capsule_nhwijc_local[(n_3 + 6)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 3)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 184)]));
              conv2d_capsule_nhwijc_local[(n_3 + 8)] = (conv2d_capsule_nhwijc_local[(n_3 + 8)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 259)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 48)]));
              conv2d_capsule_nhwijc_local[(n_3 + 10)] = (conv2d_capsule_nhwijc_local[(n_3 + 10)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 259)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 56)]));
              conv2d_capsule_nhwijc_local[(n_3 + 12)] = (conv2d_capsule_nhwijc_local[(n_3 + 12)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 259)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 176)]));
              conv2d_capsule_nhwijc_local[(n_3 + 14)] = (conv2d_capsule_nhwijc_local[(n_3 + 14)] + (PadInput_shared[((((((((n_3 * 2720) + ((((int)threadIdx.x) >> 7) * 1088)) + (rh_2 * 544)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rw_1 * 32)) + (((((int)threadIdx.x) & 31) >> 3) * 8)) + (rc_1 * 4)) + 259)] * weight_shared[(((((rh_2 * 768) + (rw_1 * 256)) + (rc_1 * 64)) + (((int)threadIdx.x) & 7)) + 184)]));
            }
          }
        }
      }
    }
  }
  conv2d_capsule_nhwijc[((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7))] = conv2d_capsule_nhwijc_local[0];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = conv2d_capsule_nhwijc_local[2];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 32)] = conv2d_capsule_nhwijc_local[4];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 40)] = conv2d_capsule_nhwijc_local[6];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 2048)] = conv2d_capsule_nhwijc_local[8];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 2056)] = conv2d_capsule_nhwijc_local[10];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 2080)] = conv2d_capsule_nhwijc_local[12];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 2088)] = conv2d_capsule_nhwijc_local[14];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 32768)] = conv2d_capsule_nhwijc_local[1];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 32776)] = conv2d_capsule_nhwijc_local[3];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 32800)] = conv2d_capsule_nhwijc_local[5];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 32808)] = conv2d_capsule_nhwijc_local[7];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 34816)] = conv2d_capsule_nhwijc_local[9];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 34824)] = conv2d_capsule_nhwijc_local[11];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 34848)] = conv2d_capsule_nhwijc_local[13];
  conv2d_capsule_nhwijc[(((((((((((int)blockIdx.x) >> 4) * 65536) + (((((int)blockIdx.x) & 15) >> 2) * 8192)) + ((((int)threadIdx.x) >> 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + (((((int)blockIdx.x) & 3) >> 1) * 64)) + ((((int)blockIdx.x) & 1) * 16)) + (((int)threadIdx.x) & 7)) + 34856)] = conv2d_capsule_nhwijc_local[15];
}

