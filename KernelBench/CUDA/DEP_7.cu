
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[7];
  __shared__ float PadInput_shared[864];
  __shared__ float placeholder_shared[144];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  PadInput_shared[((int)threadIdx.x)] = 0.000000e+00f;
  float condval;
  if (((((((int)threadIdx.x) >> 4) == 1) && (1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 32) / 48) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 32)] = condval;
  float condval_1;
  if (((((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)) < 7)) {
    condval_1 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 64) / 48) * 7168)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 7168)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 64)] = condval_1;
  float condval_2;
  if ((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)))) {
    condval_2 = placeholder[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 6144)];
  } else {
    condval_2 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 96)] = condval_2;
  float condval_3;
  if (((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_3 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 128) / 48) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval_3 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 128)] = condval_3;
  float condval_4;
  if (((((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)) < 7)) {
    condval_4 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 160) / 48) * 7168)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 7168)];
  } else {
    condval_4 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 160)] = condval_4;
  float condval_5;
  if ((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)))) {
    condval_5 = placeholder[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 20480)];
  } else {
    condval_5 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 192)] = condval_5;
  float condval_6;
  if (((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_6 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 224) / 48) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval_6 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 224)] = condval_6;
  float condval_7;
  if (((((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)) < 7)) {
    condval_7 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 256) / 48) * 7168)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 7168)];
  } else {
    condval_7 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 256)] = condval_7;
  float condval_8;
  if ((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)))) {
    condval_8 = placeholder[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 34816)];
  } else {
    condval_8 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 288)] = condval_8;
  float condval_9;
  if (((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_9 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 320) / 48) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval_9 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 320)] = condval_9;
  float condval_10;
  if (((((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)) < 7)) {
    condval_10 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 352) / 48) * 7168)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 7168)];
  } else {
    condval_10 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 352)] = condval_10;
  PadInput_shared[(((int)threadIdx.x) + 384)] = 0.000000e+00f;
  float condval_11;
  if (((((3 <= (((((int)threadIdx.x) >> 4) + 26) % 27)) && (((((int)threadIdx.x) + 416) % 432) < 384)) && (1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_11 = placeholder[((((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 416) / 432) * 50176)) + (((((((int)threadIdx.x) >> 4) + 26) % 27) / 3) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval_11 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 416)] = condval_11;
  PadInput_shared[(((int)threadIdx.x) + 448)] = 0.000000e+00f;
  float condval_12;
  if ((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)))) {
    condval_12 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 480) / 432) * 50176)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 1024)];
  } else {
    condval_12 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 480)] = condval_12;
  float condval_13;
  if (((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_13 = placeholder[((((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 512) / 432) * 50176)) + (((((int)threadIdx.x) + 80) / 48) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval_13 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 512)] = condval_13;
  float condval_14;
  if (((((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)) < 7)) {
    condval_14 = placeholder[((((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 544) / 432) * 50176)) + (((((int)threadIdx.x) + 112) / 48) * 7168)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 7168)];
  } else {
    condval_14 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 544)] = condval_14;
  float condval_15;
  if ((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)))) {
    condval_15 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 576) / 432) * 50176)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 13312)];
  } else {
    condval_15 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 576)] = condval_15;
  float condval_16;
  if (((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_16 = placeholder[((((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 608) / 432) * 50176)) + (((((int)threadIdx.x) + 176) / 48) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval_16 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 608)] = condval_16;
  float condval_17;
  if (((((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)) < 7)) {
    condval_17 = placeholder[((((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 640) / 432) * 50176)) + (((((int)threadIdx.x) + 208) / 48) * 7168)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 7168)];
  } else {
    condval_17 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 640)] = condval_17;
  float condval_18;
  if ((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)))) {
    condval_18 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 672) / 432) * 50176)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 27648)];
  } else {
    condval_18 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 672)] = condval_18;
  float condval_19;
  if (((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_19 = placeholder[((((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 704) / 432) * 50176)) + (((((int)threadIdx.x) + 272) / 48) * 7168)) + ((((((int)threadIdx.x) >> 4) + 2) % 3) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 8192)];
  } else {
    condval_19 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 704)] = condval_19;
  float condval_20;
  if (((((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)) < 7)) {
    condval_20 = placeholder[((((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 736) / 432) * 50176)) + (((((int)threadIdx.x) + 304) / 48) * 7168)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) - 7168)];
  } else {
    condval_20 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 736)] = condval_20;
  float condval_21;
  if ((1 <= (((((int)blockIdx.x) % 448) >> 6) + (((int)threadIdx.x) >> 4)))) {
    condval_21 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 768) / 432) * 50176)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 41984)];
  } else {
    condval_21 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 768)] = condval_21;
  float condval_22;
  if ((((((int)threadIdx.x) < 16) && (1 <= (((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)))) && ((((((int)blockIdx.x) % 448) >> 6) + (((((int)threadIdx.x) >> 4) + 2) % 3)) < 8))) {
    condval_22 = placeholder[(((((((((int)blockIdx.x) / 448) * 100352) + (((((int)threadIdx.x) + 800) / 432) * 50176)) + (((((int)threadIdx.x) + 368) / 48) * 7168)) + ((((int)blockIdx.x) % 448) * 16)) + ((int)threadIdx.x)) - 6144)];
  } else {
    condval_22 = 0.000000e+00f;
  }
  PadInput_shared[(((int)threadIdx.x) + 800)] = condval_22;
  PadInput_shared[(((int)threadIdx.x) + 832)] = 0.000000e+00f;
  *(float2*)(placeholder_shared + (((int)threadIdx.x) * 2)) = *(float2*)(placeholder_1 + ((((((int)threadIdx.x) >> 3) * 1024) + ((((int)blockIdx.x) & 63) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
  *(float2*)(placeholder_shared + ((((int)threadIdx.x) * 2) + 64)) = *(float2*)(placeholder_1 + (((((((int)threadIdx.x) >> 3) * 1024) + ((((int)blockIdx.x) & 63) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 4096));
  if (((int)threadIdx.x) < 8) {
    *(float2*)(placeholder_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(placeholder_1 + ((((((int)blockIdx.x) & 63) * 16) + (((int)threadIdx.x) * 2)) + 8192));
  }
  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15))] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 48)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 96)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 144)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 192)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 240)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 288)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 48)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 96)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 144)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 192)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 240)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 288)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 336)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 96)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 144)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 192)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 240)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 288)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 336)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 384)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 16)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 64)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 112)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 160)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 208)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 256)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 304)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 64)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 112)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 160)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 208)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 256)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 304)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 352)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 112)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 160)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 208)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 256)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 304)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 352)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 400)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 32)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 80)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 176)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 224)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 272)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 320)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 80)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 176)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 224)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 272)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 320)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 368)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 128)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 176)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 224)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 272)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 320)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 368)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 432) + (((int)threadIdx.x) & 15)) + 416)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 7168)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 14336)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 21504)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 28672)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 35840)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[((((((((int)blockIdx.x) / 448) * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.x) % 448) * 16)) + (((int)threadIdx.x) & 15)) + 43008)] = depth_conv2d_nhwc_local[6];
}

