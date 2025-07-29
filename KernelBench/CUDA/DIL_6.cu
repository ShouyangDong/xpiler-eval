
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
  float conv2d_nhwc_local[8];
  __shared__ float PadInput_shared[768];
  __shared__ float weight_shared[768];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  for (int rh_0 = 0; rh_0 < 3; ++rh_0) {
    for (int rc_0 = 0; rc_0 < 32; ++rc_0) {
      __syncthreads();
      float4 condval;
      if (((((1 <= (((((((int)blockIdx.x) % 96) / 24) * 3) + (rh_0 * 2)) + ((((int)threadIdx.x) % 48) >> 4))) && ((((((int)blockIdx.x) % 96) / 24) + (((rh_0 * 2) + ((((int)threadIdx.x) % 48) >> 4)) / 3)) < 5)) && (1 <= ((((((int)blockIdx.x) % 24) >> 3) * 4) + ((((int)threadIdx.x) & 15) >> 1)))) && (((((((int)blockIdx.x) % 24) >> 3) * 4) + ((((int)threadIdx.x) & 15) >> 1)) < 15))) {
        condval = *(float4*)(inputs + (((((((((((((int)blockIdx.x) / 96) * 200704) + ((((int)threadIdx.x) / 48) * 50176)) + (((((int)blockIdx.x) % 96) / 24) * 10752)) + (rh_0 * 7168)) + (((((int)threadIdx.x) % 48) >> 4) * 3584)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + (((((int)threadIdx.x) & 15) >> 1) * 256)) + (rc_0 * 8)) + ((((int)threadIdx.x) & 1) * 4)) - 3840));
      } else {
        condval = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
      }
      *(float4*)(PadInput_shared + (((int)threadIdx.x) * 4)) = condval;
      *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((rh_0 * 196608) + ((((int)threadIdx.x) >> 6) * 65536)) + (rc_0 * 2048)) + (((((int)threadIdx.x) & 63) >> 3) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
      __syncthreads();
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) >> 4) * 64)] * weight_shared[((((int)threadIdx.x) & 15) * 2)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 16)] * weight_shared[((((int)threadIdx.x) & 15) * 2)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) >> 4) * 64)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 16)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 8)] * weight_shared[((((int)threadIdx.x) & 15) * 2)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 24)] * weight_shared[((((int)threadIdx.x) & 15) * 2)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 8)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 24)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 1)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 17)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 1)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 17)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 9)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 25)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 9)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 25)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 2)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 18)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 2)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 18)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 10)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 26)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 10)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 26)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 3)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 19)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 3)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 19)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 11)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 27)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 11)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 27)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 16)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 32)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 16)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 32)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 24)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 40)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 24)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 40)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 17)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 33)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 17)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 33)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 25)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 41)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 25)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 41)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 18)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 34)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 18)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 34)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 26)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 42)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 26)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 42)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 19)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 35)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 19)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 35)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 27)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 43)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 27)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 43)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 32)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 48)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 32)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 48)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 40)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 56)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 40)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 56)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 33)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 49)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 33)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 49)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 41)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 57)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 41)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 57)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 34)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 50)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 34)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 50)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 42)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 58)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 42)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 58)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 35)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 51)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 35)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 51)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 43)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 59)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 43)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 59)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 4)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 20)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 4)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 20)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 12)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 28)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 12)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 28)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 5)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 21)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 5)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 21)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 13)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 29)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 13)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 29)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 6)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 22)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 6)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 22)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 14)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 30)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 14)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 30)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 7)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 23)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 7)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 23)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 15)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 31)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 15)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 31)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 20)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 36)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 20)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 36)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 28)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 44)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 28)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 44)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 21)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 37)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 21)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 37)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 29)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 45)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 29)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 45)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 22)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 38)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 22)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 38)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 30)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 46)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 30)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 46)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 23)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 39)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 23)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 39)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 31)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 47)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 31)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 47)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 36)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 52)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 36)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 52)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 44)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 60)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 44)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 60)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 37)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 53)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 37)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 53)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 45)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 61)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 45)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 61)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 38)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 54)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 38)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 54)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 46)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 62)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 46)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 62)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 39)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 55)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 39)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 55)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 47)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 63)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 47)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 64) + 63)] * weight_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
    }
  }
  conv2d_nhwc[((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 512)] = conv2d_nhwc_local[4];
  conv2d_nhwc[(((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 513)] = conv2d_nhwc_local[5];
  conv2d_nhwc[(((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 256)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 768)] = conv2d_nhwc_local[6];
  conv2d_nhwc[(((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 257)] = conv2d_nhwc_local[3];
  conv2d_nhwc[(((((((((((int)blockIdx.x) / 96) * 147456) + ((((int)threadIdx.x) / 48) * 36864)) + (((((int)blockIdx.x) % 96) / 24) * 9216)) + (((((int)threadIdx.x) % 48) >> 4) * 3072)) + (((((int)blockIdx.x) % 24) >> 3) * 1024)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 769)] = conv2d_nhwc_local[7];
}

