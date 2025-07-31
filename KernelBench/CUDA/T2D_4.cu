
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
  float conv2d_transpose_nhwc_local[8];
  __shared__ float PadInput_shared[256];
  __shared__ float weight_shared[4096];
  conv2d_transpose_nhwc_local[0] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[1] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[2] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[3] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[4] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[5] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[6] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[7] = 0.000000e+00f;
  for (int rw_0 = 0; rw_0 < 2; ++rw_0) {
    for (int rc_0 = 0; rc_0 < 32; ++rc_0) {
      __syncthreads();
      float condval;
      if ((((1 <= ((((((int)blockIdx.x) & 63) >> 5) * 2) + (((int)threadIdx.x) >> 5))) && (1 <= ((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0))) && (((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0) < 5))) {
        condval = inputs[((((((((((((int)blockIdx.x) >> 6) * 16384) + (((((int)blockIdx.x) & 63) >> 5) * 4096)) + ((((int)threadIdx.x) >> 5) * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + (rw_0 * 512)) + (rc_0 * 16)) + (((int)threadIdx.x) & 15)) - 2560)];
      } else {
        condval = 0.000000e+00f;
      }
      PadInput_shared[((int)threadIdx.x)] = condval;
      float condval_1;
      if ((((((((((int)blockIdx.x) & 63) >> 5) * 2) + (((int)threadIdx.x) >> 5)) < 3) && (1 <= ((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0))) && (((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0) < 5))) {
        condval_1 = inputs[((((((((((((int)blockIdx.x) >> 6) * 16384) + (((((int)blockIdx.x) & 63) >> 5) * 4096)) + ((((int)threadIdx.x) >> 5) * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + (rw_0 * 512)) + (rc_0 * 16)) + (((int)threadIdx.x) & 15)) + 1536)];
      } else {
        condval_1 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 64)] = condval_1;
      float condval_2;
      if ((((1 <= ((((((int)blockIdx.x) & 63) >> 5) * 2) + (((int)threadIdx.x) >> 5))) && (1 <= ((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0))) && (((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0) < 5))) {
        condval_2 = inputs[((((((((((((int)blockIdx.x) >> 6) * 16384) + (((((int)blockIdx.x) & 63) >> 5) * 4096)) + ((((int)threadIdx.x) >> 5) * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + (rw_0 * 512)) + (rc_0 * 16)) + (((int)threadIdx.x) & 15)) + 5632)];
      } else {
        condval_2 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 128)] = condval_2;
      float condval_3;
      if ((((((((((int)blockIdx.x) & 63) >> 5) * 2) + (((int)threadIdx.x) >> 5)) < 3) && (1 <= ((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0))) && (((((((int)threadIdx.x) & 31) >> 4) + ((((int)blockIdx.x) & 31) >> 3)) + rw_0) < 5))) {
        condval_3 = inputs[((((((((((((int)blockIdx.x) >> 6) * 16384) + (((((int)blockIdx.x) & 63) >> 5) * 4096)) + ((((int)threadIdx.x) >> 5) * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + (rw_0 * 512)) + (rc_0 * 16)) + (((int)threadIdx.x) & 15)) + 9728)];
      } else {
        condval_3 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 192)] = condval_3;
      *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 262144) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 264192) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 393216) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 395264) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 786432) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 788480) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 917504) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 919552) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1310720) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1312768) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1441792) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1443840) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1835008) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1837056) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1966080) - (rw_0 * 262144)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(weight + ((((((rc_0 * 4096) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1968128) - (rw_0 * 262144)));
      __syncthreads();
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) >> 5) * 128)] * weight_shared[((((int)threadIdx.x) & 31) + 3584)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 1)] * weight_shared[((((int)threadIdx.x) & 31) + 3616)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 2)] * weight_shared[((((int)threadIdx.x) & 31) + 3648)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 3)] * weight_shared[((((int)threadIdx.x) & 31) + 3680)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 16)] * weight_shared[((((int)threadIdx.x) & 31) + 3072)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 17)] * weight_shared[((((int)threadIdx.x) & 31) + 3104)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 18)] * weight_shared[((((int)threadIdx.x) & 31) + 3136)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 19)] * weight_shared[((((int)threadIdx.x) & 31) + 3168)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 32)] * weight_shared[((((int)threadIdx.x) & 31) + 2560)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 33)] * weight_shared[((((int)threadIdx.x) & 31) + 2592)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 34)] * weight_shared[((((int)threadIdx.x) & 31) + 2624)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 35)] * weight_shared[((((int)threadIdx.x) & 31) + 2656)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 48)] * weight_shared[((((int)threadIdx.x) & 31) + 2048)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 49)] * weight_shared[((((int)threadIdx.x) & 31) + 2080)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 50)] * weight_shared[((((int)threadIdx.x) & 31) + 2112)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 51)] * weight_shared[((((int)threadIdx.x) & 31) + 2144)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 32)] * weight_shared[((((int)threadIdx.x) & 31) + 3584)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 33)] * weight_shared[((((int)threadIdx.x) & 31) + 3616)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 34)] * weight_shared[((((int)threadIdx.x) & 31) + 3648)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 35)] * weight_shared[((((int)threadIdx.x) & 31) + 3680)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 48)] * weight_shared[((((int)threadIdx.x) & 31) + 3072)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 49)] * weight_shared[((((int)threadIdx.x) & 31) + 3104)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 50)] * weight_shared[((((int)threadIdx.x) & 31) + 3136)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 51)] * weight_shared[((((int)threadIdx.x) & 31) + 3168)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 64)] * weight_shared[((((int)threadIdx.x) & 31) + 2560)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 65)] * weight_shared[((((int)threadIdx.x) & 31) + 2592)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 66)] * weight_shared[((((int)threadIdx.x) & 31) + 2624)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 67)] * weight_shared[((((int)threadIdx.x) & 31) + 2656)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 80)] * weight_shared[((((int)threadIdx.x) & 31) + 2048)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 81)] * weight_shared[((((int)threadIdx.x) & 31) + 2080)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 82)] * weight_shared[((((int)threadIdx.x) & 31) + 2112)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 83)] * weight_shared[((((int)threadIdx.x) & 31) + 2144)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 4)] * weight_shared[((((int)threadIdx.x) & 31) + 3712)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 5)] * weight_shared[((((int)threadIdx.x) & 31) + 3744)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 6)] * weight_shared[((((int)threadIdx.x) & 31) + 3776)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 7)] * weight_shared[((((int)threadIdx.x) & 31) + 3808)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 20)] * weight_shared[((((int)threadIdx.x) & 31) + 3200)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 21)] * weight_shared[((((int)threadIdx.x) & 31) + 3232)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 22)] * weight_shared[((((int)threadIdx.x) & 31) + 3264)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 23)] * weight_shared[((((int)threadIdx.x) & 31) + 3296)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 36)] * weight_shared[((((int)threadIdx.x) & 31) + 2688)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 37)] * weight_shared[((((int)threadIdx.x) & 31) + 2720)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 38)] * weight_shared[((((int)threadIdx.x) & 31) + 2752)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 39)] * weight_shared[((((int)threadIdx.x) & 31) + 2784)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 52)] * weight_shared[((((int)threadIdx.x) & 31) + 2176)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 53)] * weight_shared[((((int)threadIdx.x) & 31) + 2208)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 54)] * weight_shared[((((int)threadIdx.x) & 31) + 2240)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 55)] * weight_shared[((((int)threadIdx.x) & 31) + 2272)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 36)] * weight_shared[((((int)threadIdx.x) & 31) + 3712)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 37)] * weight_shared[((((int)threadIdx.x) & 31) + 3744)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 38)] * weight_shared[((((int)threadIdx.x) & 31) + 3776)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 39)] * weight_shared[((((int)threadIdx.x) & 31) + 3808)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 52)] * weight_shared[((((int)threadIdx.x) & 31) + 3200)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 53)] * weight_shared[((((int)threadIdx.x) & 31) + 3232)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 54)] * weight_shared[((((int)threadIdx.x) & 31) + 3264)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 55)] * weight_shared[((((int)threadIdx.x) & 31) + 3296)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 68)] * weight_shared[((((int)threadIdx.x) & 31) + 2688)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 69)] * weight_shared[((((int)threadIdx.x) & 31) + 2720)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 70)] * weight_shared[((((int)threadIdx.x) & 31) + 2752)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 71)] * weight_shared[((((int)threadIdx.x) & 31) + 2784)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 84)] * weight_shared[((((int)threadIdx.x) & 31) + 2176)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 85)] * weight_shared[((((int)threadIdx.x) & 31) + 2208)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 86)] * weight_shared[((((int)threadIdx.x) & 31) + 2240)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 87)] * weight_shared[((((int)threadIdx.x) & 31) + 2272)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 8)] * weight_shared[((((int)threadIdx.x) & 31) + 3840)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 9)] * weight_shared[((((int)threadIdx.x) & 31) + 3872)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 10)] * weight_shared[((((int)threadIdx.x) & 31) + 3904)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 11)] * weight_shared[((((int)threadIdx.x) & 31) + 3936)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 24)] * weight_shared[((((int)threadIdx.x) & 31) + 3328)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 25)] * weight_shared[((((int)threadIdx.x) & 31) + 3360)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 26)] * weight_shared[((((int)threadIdx.x) & 31) + 3392)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 27)] * weight_shared[((((int)threadIdx.x) & 31) + 3424)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 40)] * weight_shared[((((int)threadIdx.x) & 31) + 2816)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 41)] * weight_shared[((((int)threadIdx.x) & 31) + 2848)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 42)] * weight_shared[((((int)threadIdx.x) & 31) + 2880)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 43)] * weight_shared[((((int)threadIdx.x) & 31) + 2912)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 56)] * weight_shared[((((int)threadIdx.x) & 31) + 2304)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 57)] * weight_shared[((((int)threadIdx.x) & 31) + 2336)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 58)] * weight_shared[((((int)threadIdx.x) & 31) + 2368)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 59)] * weight_shared[((((int)threadIdx.x) & 31) + 2400)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 40)] * weight_shared[((((int)threadIdx.x) & 31) + 3840)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 41)] * weight_shared[((((int)threadIdx.x) & 31) + 3872)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 42)] * weight_shared[((((int)threadIdx.x) & 31) + 3904)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 43)] * weight_shared[((((int)threadIdx.x) & 31) + 3936)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 56)] * weight_shared[((((int)threadIdx.x) & 31) + 3328)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 57)] * weight_shared[((((int)threadIdx.x) & 31) + 3360)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 58)] * weight_shared[((((int)threadIdx.x) & 31) + 3392)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 59)] * weight_shared[((((int)threadIdx.x) & 31) + 3424)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 72)] * weight_shared[((((int)threadIdx.x) & 31) + 2816)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 73)] * weight_shared[((((int)threadIdx.x) & 31) + 2848)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 74)] * weight_shared[((((int)threadIdx.x) & 31) + 2880)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 75)] * weight_shared[((((int)threadIdx.x) & 31) + 2912)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 88)] * weight_shared[((((int)threadIdx.x) & 31) + 2304)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 89)] * weight_shared[((((int)threadIdx.x) & 31) + 2336)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 90)] * weight_shared[((((int)threadIdx.x) & 31) + 2368)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 91)] * weight_shared[((((int)threadIdx.x) & 31) + 2400)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 12)] * weight_shared[((((int)threadIdx.x) & 31) + 3968)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 13)] * weight_shared[((((int)threadIdx.x) & 31) + 4000)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 14)] * weight_shared[((((int)threadIdx.x) & 31) + 4032)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 15)] * weight_shared[((((int)threadIdx.x) & 31) + 4064)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 28)] * weight_shared[((((int)threadIdx.x) & 31) + 3456)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 29)] * weight_shared[((((int)threadIdx.x) & 31) + 3488)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 30)] * weight_shared[((((int)threadIdx.x) & 31) + 3520)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 31)] * weight_shared[((((int)threadIdx.x) & 31) + 3552)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 44)] * weight_shared[((((int)threadIdx.x) & 31) + 2944)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 45)] * weight_shared[((((int)threadIdx.x) & 31) + 2976)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 46)] * weight_shared[((((int)threadIdx.x) & 31) + 3008)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 47)] * weight_shared[((((int)threadIdx.x) & 31) + 3040)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 60)] * weight_shared[((((int)threadIdx.x) & 31) + 2432)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 61)] * weight_shared[((((int)threadIdx.x) & 31) + 2464)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 62)] * weight_shared[((((int)threadIdx.x) & 31) + 2496)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 63)] * weight_shared[((((int)threadIdx.x) & 31) + 2528)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 44)] * weight_shared[((((int)threadIdx.x) & 31) + 3968)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 45)] * weight_shared[((((int)threadIdx.x) & 31) + 4000)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 46)] * weight_shared[((((int)threadIdx.x) & 31) + 4032)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 47)] * weight_shared[((((int)threadIdx.x) & 31) + 4064)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 60)] * weight_shared[((((int)threadIdx.x) & 31) + 3456)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 61)] * weight_shared[((((int)threadIdx.x) & 31) + 3488)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 62)] * weight_shared[((((int)threadIdx.x) & 31) + 3520)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 63)] * weight_shared[((((int)threadIdx.x) & 31) + 3552)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 76)] * weight_shared[((((int)threadIdx.x) & 31) + 2944)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 77)] * weight_shared[((((int)threadIdx.x) & 31) + 2976)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 78)] * weight_shared[((((int)threadIdx.x) & 31) + 3008)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 79)] * weight_shared[((((int)threadIdx.x) & 31) + 3040)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 92)] * weight_shared[((((int)threadIdx.x) & 31) + 2432)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 93)] * weight_shared[((((int)threadIdx.x) & 31) + 2464)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 94)] * weight_shared[((((int)threadIdx.x) & 31) + 2496)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 95)] * weight_shared[((((int)threadIdx.x) & 31) + 2528)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 32)] * weight_shared[((((int)threadIdx.x) & 31) + 1536)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 33)] * weight_shared[((((int)threadIdx.x) & 31) + 1568)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 34)] * weight_shared[((((int)threadIdx.x) & 31) + 1600)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 35)] * weight_shared[((((int)threadIdx.x) & 31) + 1632)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 48)] * weight_shared[((((int)threadIdx.x) & 31) + 1024)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 49)] * weight_shared[((((int)threadIdx.x) & 31) + 1056)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 50)] * weight_shared[((((int)threadIdx.x) & 31) + 1088)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 51)] * weight_shared[((((int)threadIdx.x) & 31) + 1120)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 64)] * weight_shared[((((int)threadIdx.x) & 31) + 512)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 65)] * weight_shared[((((int)threadIdx.x) & 31) + 544)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 66)] * weight_shared[((((int)threadIdx.x) & 31) + 576)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 67)] * weight_shared[((((int)threadIdx.x) & 31) + 608)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 80)] * weight_shared[(((int)threadIdx.x) & 31)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 81)] * weight_shared[((((int)threadIdx.x) & 31) + 32)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 82)] * weight_shared[((((int)threadIdx.x) & 31) + 64)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 83)] * weight_shared[((((int)threadIdx.x) & 31) + 96)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 64)] * weight_shared[((((int)threadIdx.x) & 31) + 1536)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 65)] * weight_shared[((((int)threadIdx.x) & 31) + 1568)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 66)] * weight_shared[((((int)threadIdx.x) & 31) + 1600)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 67)] * weight_shared[((((int)threadIdx.x) & 31) + 1632)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 80)] * weight_shared[((((int)threadIdx.x) & 31) + 1024)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 81)] * weight_shared[((((int)threadIdx.x) & 31) + 1056)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 82)] * weight_shared[((((int)threadIdx.x) & 31) + 1088)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 83)] * weight_shared[((((int)threadIdx.x) & 31) + 1120)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 96)] * weight_shared[((((int)threadIdx.x) & 31) + 512)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 97)] * weight_shared[((((int)threadIdx.x) & 31) + 544)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 98)] * weight_shared[((((int)threadIdx.x) & 31) + 576)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 99)] * weight_shared[((((int)threadIdx.x) & 31) + 608)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 112)] * weight_shared[(((int)threadIdx.x) & 31)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 113)] * weight_shared[((((int)threadIdx.x) & 31) + 32)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 114)] * weight_shared[((((int)threadIdx.x) & 31) + 64)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 115)] * weight_shared[((((int)threadIdx.x) & 31) + 96)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 36)] * weight_shared[((((int)threadIdx.x) & 31) + 1664)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 37)] * weight_shared[((((int)threadIdx.x) & 31) + 1696)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 38)] * weight_shared[((((int)threadIdx.x) & 31) + 1728)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 39)] * weight_shared[((((int)threadIdx.x) & 31) + 1760)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 52)] * weight_shared[((((int)threadIdx.x) & 31) + 1152)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 53)] * weight_shared[((((int)threadIdx.x) & 31) + 1184)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 54)] * weight_shared[((((int)threadIdx.x) & 31) + 1216)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 55)] * weight_shared[((((int)threadIdx.x) & 31) + 1248)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 68)] * weight_shared[((((int)threadIdx.x) & 31) + 640)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 69)] * weight_shared[((((int)threadIdx.x) & 31) + 672)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 70)] * weight_shared[((((int)threadIdx.x) & 31) + 704)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 71)] * weight_shared[((((int)threadIdx.x) & 31) + 736)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 84)] * weight_shared[((((int)threadIdx.x) & 31) + 128)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 85)] * weight_shared[((((int)threadIdx.x) & 31) + 160)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 86)] * weight_shared[((((int)threadIdx.x) & 31) + 192)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 87)] * weight_shared[((((int)threadIdx.x) & 31) + 224)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 68)] * weight_shared[((((int)threadIdx.x) & 31) + 1664)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 69)] * weight_shared[((((int)threadIdx.x) & 31) + 1696)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 70)] * weight_shared[((((int)threadIdx.x) & 31) + 1728)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 71)] * weight_shared[((((int)threadIdx.x) & 31) + 1760)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 84)] * weight_shared[((((int)threadIdx.x) & 31) + 1152)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 85)] * weight_shared[((((int)threadIdx.x) & 31) + 1184)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 86)] * weight_shared[((((int)threadIdx.x) & 31) + 1216)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 87)] * weight_shared[((((int)threadIdx.x) & 31) + 1248)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 100)] * weight_shared[((((int)threadIdx.x) & 31) + 640)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 101)] * weight_shared[((((int)threadIdx.x) & 31) + 672)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 102)] * weight_shared[((((int)threadIdx.x) & 31) + 704)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 103)] * weight_shared[((((int)threadIdx.x) & 31) + 736)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 116)] * weight_shared[((((int)threadIdx.x) & 31) + 128)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 117)] * weight_shared[((((int)threadIdx.x) & 31) + 160)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 118)] * weight_shared[((((int)threadIdx.x) & 31) + 192)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 119)] * weight_shared[((((int)threadIdx.x) & 31) + 224)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 40)] * weight_shared[((((int)threadIdx.x) & 31) + 1792)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 41)] * weight_shared[((((int)threadIdx.x) & 31) + 1824)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 42)] * weight_shared[((((int)threadIdx.x) & 31) + 1856)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 43)] * weight_shared[((((int)threadIdx.x) & 31) + 1888)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 56)] * weight_shared[((((int)threadIdx.x) & 31) + 1280)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 57)] * weight_shared[((((int)threadIdx.x) & 31) + 1312)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 58)] * weight_shared[((((int)threadIdx.x) & 31) + 1344)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 59)] * weight_shared[((((int)threadIdx.x) & 31) + 1376)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 72)] * weight_shared[((((int)threadIdx.x) & 31) + 768)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 73)] * weight_shared[((((int)threadIdx.x) & 31) + 800)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 74)] * weight_shared[((((int)threadIdx.x) & 31) + 832)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 75)] * weight_shared[((((int)threadIdx.x) & 31) + 864)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 88)] * weight_shared[((((int)threadIdx.x) & 31) + 256)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 89)] * weight_shared[((((int)threadIdx.x) & 31) + 288)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 90)] * weight_shared[((((int)threadIdx.x) & 31) + 320)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 91)] * weight_shared[((((int)threadIdx.x) & 31) + 352)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 72)] * weight_shared[((((int)threadIdx.x) & 31) + 1792)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 73)] * weight_shared[((((int)threadIdx.x) & 31) + 1824)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 74)] * weight_shared[((((int)threadIdx.x) & 31) + 1856)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 75)] * weight_shared[((((int)threadIdx.x) & 31) + 1888)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 88)] * weight_shared[((((int)threadIdx.x) & 31) + 1280)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 89)] * weight_shared[((((int)threadIdx.x) & 31) + 1312)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 90)] * weight_shared[((((int)threadIdx.x) & 31) + 1344)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 91)] * weight_shared[((((int)threadIdx.x) & 31) + 1376)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 104)] * weight_shared[((((int)threadIdx.x) & 31) + 768)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 105)] * weight_shared[((((int)threadIdx.x) & 31) + 800)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 106)] * weight_shared[((((int)threadIdx.x) & 31) + 832)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 107)] * weight_shared[((((int)threadIdx.x) & 31) + 864)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 120)] * weight_shared[((((int)threadIdx.x) & 31) + 256)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 121)] * weight_shared[((((int)threadIdx.x) & 31) + 288)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 122)] * weight_shared[((((int)threadIdx.x) & 31) + 320)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 123)] * weight_shared[((((int)threadIdx.x) & 31) + 352)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 44)] * weight_shared[((((int)threadIdx.x) & 31) + 1920)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 45)] * weight_shared[((((int)threadIdx.x) & 31) + 1952)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 46)] * weight_shared[((((int)threadIdx.x) & 31) + 1984)]));
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 47)] * weight_shared[((((int)threadIdx.x) & 31) + 2016)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 60)] * weight_shared[((((int)threadIdx.x) & 31) + 1408)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 61)] * weight_shared[((((int)threadIdx.x) & 31) + 1440)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 62)] * weight_shared[((((int)threadIdx.x) & 31) + 1472)]));
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 63)] * weight_shared[((((int)threadIdx.x) & 31) + 1504)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 76)] * weight_shared[((((int)threadIdx.x) & 31) + 896)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 77)] * weight_shared[((((int)threadIdx.x) & 31) + 928)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 78)] * weight_shared[((((int)threadIdx.x) & 31) + 960)]));
      conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 79)] * weight_shared[((((int)threadIdx.x) & 31) + 992)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 92)] * weight_shared[((((int)threadIdx.x) & 31) + 384)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 93)] * weight_shared[((((int)threadIdx.x) & 31) + 416)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 94)] * weight_shared[((((int)threadIdx.x) & 31) + 448)]));
      conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 95)] * weight_shared[((((int)threadIdx.x) & 31) + 480)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 76)] * weight_shared[((((int)threadIdx.x) & 31) + 1920)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 77)] * weight_shared[((((int)threadIdx.x) & 31) + 1952)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 78)] * weight_shared[((((int)threadIdx.x) & 31) + 1984)]));
      conv2d_transpose_nhwc_local[4] = (conv2d_transpose_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 79)] * weight_shared[((((int)threadIdx.x) & 31) + 2016)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 92)] * weight_shared[((((int)threadIdx.x) & 31) + 1408)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 93)] * weight_shared[((((int)threadIdx.x) & 31) + 1440)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 94)] * weight_shared[((((int)threadIdx.x) & 31) + 1472)]));
      conv2d_transpose_nhwc_local[5] = (conv2d_transpose_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 95)] * weight_shared[((((int)threadIdx.x) & 31) + 1504)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 108)] * weight_shared[((((int)threadIdx.x) & 31) + 896)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 109)] * weight_shared[((((int)threadIdx.x) & 31) + 928)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 110)] * weight_shared[((((int)threadIdx.x) & 31) + 960)]));
      conv2d_transpose_nhwc_local[6] = (conv2d_transpose_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 111)] * weight_shared[((((int)threadIdx.x) & 31) + 992)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 124)] * weight_shared[((((int)threadIdx.x) & 31) + 384)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 125)] * weight_shared[((((int)threadIdx.x) & 31) + 416)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 126)] * weight_shared[((((int)threadIdx.x) & 31) + 448)]));
      conv2d_transpose_nhwc_local[7] = (conv2d_transpose_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + 127)] * weight_shared[((((int)threadIdx.x) & 31) + 480)]));
    }
  }
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31)) + 256)] = conv2d_transpose_nhwc_local[1];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31)) + 2048)] = conv2d_transpose_nhwc_local[2];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31)) + 2304)] = conv2d_transpose_nhwc_local[3];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31)) + 4096)] = conv2d_transpose_nhwc_local[4];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31)) + 4352)] = conv2d_transpose_nhwc_local[5];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31)) + 6144)] = conv2d_transpose_nhwc_local[6];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 5) * 16384)) + (((((int)blockIdx.x) & 63) >> 5) * 8192)) + (((((int)blockIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + (((int)threadIdx.x) & 31)) + 6400)] = conv2d_transpose_nhwc_local[7];
}

