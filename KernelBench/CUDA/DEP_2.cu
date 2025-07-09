
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
extern "C" __global__ void __launch_bounds__(392) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(392) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[2];
  __shared__ float PadInput_shared[3600];
  __shared__ float placeholder_shared[144];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  float condval;
  if (((60 <= ((int)threadIdx.x)) && (4 <= (((int)threadIdx.x) % 60)))) {
    condval = placeholder[((((((((int)threadIdx.x) / 60) * 7168) + (((((int)threadIdx.x) % 60) >> 2) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7680)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) * 4)] = condval;
  float condval_1;
  if (((60 <= ((int)threadIdx.x)) && (4 <= (((int)threadIdx.x) % 60)))) {
    condval_1 = placeholder[((((((((int)threadIdx.x) / 60) * 7168) + (((((int)threadIdx.x) % 60) >> 2) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7679)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 4) + 1)] = condval_1;
  float condval_2;
  if (((60 <= ((int)threadIdx.x)) && (4 <= (((int)threadIdx.x) % 60)))) {
    condval_2 = placeholder[((((((((int)threadIdx.x) / 60) * 7168) + (((((int)threadIdx.x) % 60) >> 2) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7678)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 4) + 2)] = condval_2;
  float condval_3;
  if (((60 <= ((int)threadIdx.x)) && (4 <= (((int)threadIdx.x) % 60)))) {
    condval_3 = placeholder[((((((((int)threadIdx.x) / 60) * 7168) + (((((int)threadIdx.x) % 60) >> 2) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7677)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 4) + 3)] = condval_3;
  float condval_4;
  if ((1 <= (((((int)threadIdx.x) >> 2) + 8) % 15))) {
    condval_4 = placeholder[(((((((((int)threadIdx.x) + 392) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 8) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7680)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 4) + 1568)] = condval_4;
  float condval_5;
  if ((1 <= (((((int)threadIdx.x) >> 2) + 8) % 15))) {
    condval_5 = placeholder[(((((((((int)threadIdx.x) + 392) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 8) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7679)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 4) + 1569)] = condval_5;
  float condval_6;
  if ((1 <= (((((int)threadIdx.x) >> 2) + 8) % 15))) {
    condval_6 = placeholder[(((((((((int)threadIdx.x) + 392) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 8) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7678)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 4) + 1570)] = condval_6;
  float condval_7;
  if ((1 <= (((((int)threadIdx.x) >> 2) + 8) % 15))) {
    condval_7 = placeholder[(((((((((int)threadIdx.x) + 392) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 8) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7677)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[((((int)threadIdx.x) * 4) + 1571)] = condval_7;
  if (((int)threadIdx.x) < 116) {
    float condval_8;
    if ((1 <= (((((int)threadIdx.x) >> 2) + 1) % 15))) {
      condval_8 = placeholder[(((((((((int)threadIdx.x) + 784) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 1) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7680)];
    } else {
      condval_8 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 4) + 3136)] = condval_8;
    float condval_9;
    if ((1 <= (((((int)threadIdx.x) >> 2) + 1) % 15))) {
      condval_9 = placeholder[(((((((((int)threadIdx.x) + 784) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 1) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7679)];
    } else {
      condval_9 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 4) + 3137)] = condval_9;
    float condval_10;
    if ((1 <= (((((int)threadIdx.x) >> 2) + 1) % 15))) {
      condval_10 = placeholder[(((((((((int)threadIdx.x) + 784) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 1) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7678)];
    } else {
      condval_10 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 4) + 3138)] = condval_10;
    float condval_11;
    if ((1 <= (((((int)threadIdx.x) >> 2) + 1) % 15))) {
      condval_11 = placeholder[(((((((((int)threadIdx.x) + 784) / 60) * 7168) + ((((((int)threadIdx.x) >> 2) + 1) % 15) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7677)];
    } else {
      condval_11 = 0.000000e+00f;
    }
    PadInput_shared[((((int)threadIdx.x) * 4) + 3139)] = condval_11;
  }
  if (((int)threadIdx.x) < 36) {
    *(float4*)(placeholder_shared + (((int)threadIdx.x) * 4)) = *(float4*)(placeholder_1 + ((((((int)threadIdx.x) >> 2) * 512) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  }
  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2))] * placeholder_shared[((((int)threadIdx.x) & 7) * 2)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 1)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 16)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 17)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 32)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 33)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 240)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 241)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 256)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 257)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 272)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 273)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 480)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 481)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 496)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 497)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 512)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 56) * 480) + (((((int)threadIdx.x) % 56) >> 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 513)] * placeholder_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  depth_conv2d_nhwc[((((((int)threadIdx.x) >> 3) * 512) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[(((((((int)threadIdx.x) >> 3) * 512) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1)] = depth_conv2d_nhwc_local[1];
}

