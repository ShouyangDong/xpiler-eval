
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
  float conv2d_transpose_nhwc_local[4];
  __shared__ float PadInput_shared[1536];
  __shared__ float weight_shared[8192];
  conv2d_transpose_nhwc_local[0] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[1] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[2] = 0.000000e+00f;
  conv2d_transpose_nhwc_local[3] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 8; ++rc_0) {
    __syncthreads();
    PadInput_shared[((int)threadIdx.x)] = 0.000000e+00f;
    float condval;
    if (((((int)blockIdx.x) >> 5) == 1)) {
      condval = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) - 2048)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 64)] = condval;
    float condval_1;
    if (((((int)blockIdx.x) >> 5) == 1)) {
      condval_1 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) - 1536)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 128)] = condval_1;
    float condval_2;
    if (((((int)blockIdx.x) >> 5) == 1)) {
      condval_2 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) - 1024)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 192)] = condval_2;
    float condval_3;
    if (((((int)blockIdx.x) >> 5) == 1)) {
      condval_3 = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) - 512)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 256)] = condval_3;
    PadInput_shared[(((int)threadIdx.x) + 320)] = 0.000000e+00f;
    PadInput_shared[(((int)threadIdx.x) + 384)] = 0.000000e+00f;
    PadInput_shared[(((int)threadIdx.x) + 448)] = inputs[((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x))];
    PadInput_shared[(((int)threadIdx.x) + 512)] = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) + 512)];
    PadInput_shared[(((int)threadIdx.x) + 576)] = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) + 1024)];
    PadInput_shared[(((int)threadIdx.x) + 640)] = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) + 1536)];
    PadInput_shared[(((int)threadIdx.x) + 704)] = 0.000000e+00f;
    PadInput_shared[(((int)threadIdx.x) + 768)] = 0.000000e+00f;
    PadInput_shared[(((int)threadIdx.x) + 832)] = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) + 2048)];
    PadInput_shared[(((int)threadIdx.x) + 896)] = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) + 2560)];
    PadInput_shared[(((int)threadIdx.x) + 960)] = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) + 3072)];
    PadInput_shared[(((int)threadIdx.x) + 1024)] = inputs[(((((((int)blockIdx.x) >> 5) * 4096) + (rc_0 * 64)) + ((int)threadIdx.x)) + 3584)];
    PadInput_shared[(((int)threadIdx.x) + 1088)] = 0.000000e+00f;
    PadInput_shared[(((int)threadIdx.x) + 1152)] = 0.000000e+00f;
    float condval_4;
    if ((((int)blockIdx.x) < 32)) {
      condval_4 = inputs[(((rc_0 * 64) + ((int)threadIdx.x)) + 4096)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1216)] = condval_4;
    float condval_5;
    if ((((int)blockIdx.x) < 32)) {
      condval_5 = inputs[(((rc_0 * 64) + ((int)threadIdx.x)) + 4608)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1280)] = condval_5;
    float condval_6;
    if ((((int)blockIdx.x) < 32)) {
      condval_6 = inputs[(((rc_0 * 64) + ((int)threadIdx.x)) + 5120)];
    } else {
      condval_6 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1344)] = condval_6;
    float condval_7;
    if ((((int)blockIdx.x) < 32)) {
      condval_7 = inputs[(((rc_0 * 64) + ((int)threadIdx.x)) + 5632)];
    } else {
      condval_7 = 0.000000e+00f;
    }
    PadInput_shared[(((int)threadIdx.x) + 1408)] = condval_7;
    PadInput_shared[(((int)threadIdx.x) + 1472)] = 0.000000e+00f;
    *(float2*)(weight_shared + (((int)threadIdx.x) * 2)) = *(float2*)(weight + ((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 384)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12288));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 131072));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 640)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 135168));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 139264));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 896)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 143360));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 262144));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1152)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 266240));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1280)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 270336));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1408)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 274432));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1536)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 393216));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1664)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 397312));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1792)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 401408));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 1920)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 405504));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2048)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 524288));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2176)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 528384));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2304)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 532480));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2432)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 536576));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2560)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 655360));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2688)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 659456));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2816)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 663552));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 2944)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 667648));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3072)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 786432));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3200)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 790528));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3328)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 794624));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3456)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 798720));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3584)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 917504));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3712)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 921600));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3840)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 925696));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 3968)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 929792));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4096)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1048576));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4224)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1052672));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4352)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1056768));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4480)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1060864));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4608)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1179648));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4736)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1183744));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4864)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1187840));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 4992)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1191936));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 5120)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1310720));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 5248)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1314816));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 5376)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1318912));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 5504)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1323008));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 5632)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1441792));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 5760)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1445888));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 5888)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1449984));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6016)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1454080));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6144)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1572864));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6272)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1576960));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6400)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1581056));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6528)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1585152));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6656)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1703936));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6784)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1708032));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 6912)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1712128));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7040)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1716224));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7168)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1835008));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7296)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1839104));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7424)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1843200));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7552)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1847296));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7680)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1966080));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7808)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1970176));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 7936)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1974272));
    *(float2*)(weight_shared + ((((int)threadIdx.x) * 2) + 8064)) = *(float2*)(weight + (((((rc_0 * 16384) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 31) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1978368));
    __syncthreads();
    for (int rh_1 = 0; rh_1 < 2; ++rh_1) {
      for (int rc_1 = 0; rc_1 < 4; ++rc_1) {
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16))] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7680) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 1)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7688) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 2)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7696) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 3)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7704) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 4)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7712) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 5)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7720) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 6)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7728) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 7)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7736) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 8)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7744) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 9)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7752) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 10)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7760) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 11)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7768) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 12)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7776) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 13)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7784) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 14)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7792) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 15)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7800) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 64)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6656) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 65)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6664) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 66)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6672) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 67)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6680) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 68)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6688) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 69)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6696) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 70)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6704) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 71)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6712) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 72)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6720) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 73)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6728) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 74)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6736) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 75)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6744) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 76)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6752) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 77)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6760) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 78)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6768) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 79)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6776) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 64)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7168) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 65)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7176) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 66)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7184) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 67)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7192) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 68)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7200) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 69)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7208) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 70)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7216) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 71)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7224) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 72)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7232) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 73)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7240) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 74)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7248) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 75)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7256) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 76)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7264) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 77)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7272) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 78)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7280) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 79)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 7288) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 128)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6144) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 129)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6152) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 130)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6160) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 131)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6168) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 132)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6176) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 133)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6184) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 134)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6192) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 135)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6200) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 136)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6208) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 137)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6216) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 138)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6224) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 139)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6232) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 140)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6240) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 141)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6248) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 142)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6256) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 143)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 6264) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 384)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5632) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 385)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5640) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 386)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5648) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 387)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5656) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 388)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5664) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 389)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5672) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 390)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5680) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 391)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5688) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 392)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5696) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 393)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5704) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 394)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5712) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 395)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5720) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 396)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5728) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 397)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5736) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 398)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5744) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 399)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5752) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 448)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4608) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 449)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4616) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 450)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4624) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 451)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4632) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 452)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4640) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 453)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4648) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 454)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4656) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 455)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4664) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 456)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4672) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 457)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4680) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 458)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4688) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 459)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4696) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 460)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4704) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 461)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4712) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 462)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4720) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 463)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4728) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 448)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5120) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 449)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5128) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 450)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5136) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 451)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5144) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 452)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5152) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 453)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5160) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 454)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5168) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 455)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5176) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 456)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5184) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 457)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5192) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 458)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5200) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 459)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5208) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 460)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5216) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 461)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5224) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 462)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5232) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 463)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 5240) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 512)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4096) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 513)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4104) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 514)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4112) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 515)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4120) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 516)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4128) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 517)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4136) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 518)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4144) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 519)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4152) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 520)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4160) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 521)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4168) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 522)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4176) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 523)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4184) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 524)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4192) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 525)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4200) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 526)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4208) - (rh_1 * 4096))]));
        conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[((((((((int)threadIdx.x) >> 5) * 384) + (rh_1 * 384)) + (((((int)threadIdx.x) & 31) >> 3) * 64)) + (rc_1 * 16)) + 527)] * weight_shared[((((rc_1 * 128) + (((int)threadIdx.x) & 7)) + 4216) - (rh_1 * 4096))]));
      }
    }
  }
  conv2d_transpose_nhwc[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7)) + 256)] = conv2d_transpose_nhwc_local[1];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7)) + 2048)] = conv2d_transpose_nhwc_local[2];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 512)) + ((((int)blockIdx.x) & 31) * 8)) + (((int)threadIdx.x) & 7)) + 2304)] = conv2d_transpose_nhwc_local[3];
}

