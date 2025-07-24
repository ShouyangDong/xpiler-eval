
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[7];
  __shared__ float PadInput_shared[144];
  __shared__ float weight_shared[1536];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  for (int rh_0 = 0; rh_0 < 3; ++rh_0) {
    for (int rc_0 = 0; rc_0 < 8; ++rc_0) {
      __syncthreads();
      float4 condval;
      if ((((1 <= (((((int)blockIdx.x) % 112) >> 4) + rh_0)) && ((((((int)blockIdx.x) % 112) >> 4) + rh_0) < 8)) && (4 <= ((int)threadIdx.x)))) {
        condval = *(float4*)(inputs + ((((((((((int)blockIdx.x) >> 4) * 3584) + (rh_0 * 3584)) + ((((int)threadIdx.x) >> 2) * 512)) + (((((int)blockIdx.x) & 15) >> 2) * 128)) + (rc_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 4096));
      } else {
        condval = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
      }
      *(float4*)(PadInput_shared + (((int)threadIdx.x) * 4)) = condval;
      if (((int)threadIdx.x) < 4) {
        *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 128)) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
      }
      *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + (((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2048));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 4096));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 6144));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 65536));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 640)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 67584));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 69632));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 71680));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 131072));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1152)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 133120));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 135168));
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1408)) = *(float4*)(weight + ((((((rh_0 * 196608) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 137216));
      __syncthreads();
      for (int rw_1 = 0; rw_1 < 3; ++rw_1) {
        for (int rc_1 = 0; rc_1 < 8; ++rc_1) {
          conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((rw_1 * 16) + (rc_1 * 2))] * weight_shared[(((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x))]));
          conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 16)] * weight_shared[(((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x))]));
          conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 32)] * weight_shared[(((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x))]));
          conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 48)] * weight_shared[(((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x))]));
          conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 64)] * weight_shared[(((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x))]));
          conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 80)] * weight_shared[(((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x))]));
          conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 96)] * weight_shared[(((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x))]));
          conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 1)] * weight_shared[((((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x)) + 32)]));
          conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 17)] * weight_shared[((((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x)) + 32)]));
          conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 33)] * weight_shared[((((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x)) + 32)]));
          conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 49)] * weight_shared[((((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x)) + 32)]));
          conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 65)] * weight_shared[((((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x)) + 32)]));
          conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 81)] * weight_shared[((((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x)) + 32)]));
          conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((rw_1 * 16) + (rc_1 * 2)) + 97)] * weight_shared[((((rw_1 * 512) + (rc_1 * 64)) + ((int)threadIdx.x)) + 32)]));
        }
      }
    }
  }
  conv2d_nhwc[((((((int)blockIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((int)blockIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x)) + 512)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((int)blockIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x)) + 1024)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((int)blockIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x)) + 1536)] = conv2d_nhwc_local[3];
  conv2d_nhwc[(((((((int)blockIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x)) + 2048)] = conv2d_nhwc_local[4];
  conv2d_nhwc[(((((((int)blockIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x)) + 2560)] = conv2d_nhwc_local[5];
  conv2d_nhwc[(((((((int)blockIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x)) + 3072)] = conv2d_nhwc_local[6];
}

