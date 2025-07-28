
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
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[32];
  __shared__ float PadInput_shared[4096];
  __shared__ float weight_shared[2048];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[8] = 0.000000e+00f;
  conv2d_nhwc_local[10] = 0.000000e+00f;
  conv2d_nhwc_local[12] = 0.000000e+00f;
  conv2d_nhwc_local[14] = 0.000000e+00f;
  conv2d_nhwc_local[16] = 0.000000e+00f;
  conv2d_nhwc_local[18] = 0.000000e+00f;
  conv2d_nhwc_local[20] = 0.000000e+00f;
  conv2d_nhwc_local[22] = 0.000000e+00f;
  conv2d_nhwc_local[24] = 0.000000e+00f;
  conv2d_nhwc_local[26] = 0.000000e+00f;
  conv2d_nhwc_local[28] = 0.000000e+00f;
  conv2d_nhwc_local[30] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  conv2d_nhwc_local[9] = 0.000000e+00f;
  conv2d_nhwc_local[11] = 0.000000e+00f;
  conv2d_nhwc_local[13] = 0.000000e+00f;
  conv2d_nhwc_local[15] = 0.000000e+00f;
  conv2d_nhwc_local[17] = 0.000000e+00f;
  conv2d_nhwc_local[19] = 0.000000e+00f;
  conv2d_nhwc_local[21] = 0.000000e+00f;
  conv2d_nhwc_local[23] = 0.000000e+00f;
  conv2d_nhwc_local[25] = 0.000000e+00f;
  conv2d_nhwc_local[27] = 0.000000e+00f;
  conv2d_nhwc_local[29] = 0.000000e+00f;
  conv2d_nhwc_local[31] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 2; ++rc_0) {
    __syncthreads();
    PadInput_shared[((int)threadIdx.x)] = inputs[(((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31))];
    PadInput_shared[(((int)threadIdx.x) + 256)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    PadInput_shared[(((int)threadIdx.x) + 512)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 200704)];
    PadInput_shared[(((int)threadIdx.x) + 768)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 207872)];
    PadInput_shared[(((int)threadIdx.x) + 1024)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 401408)];
    PadInput_shared[(((int)threadIdx.x) + 1280)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 408576)];
    PadInput_shared[(((int)threadIdx.x) + 1536)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 602112)];
    PadInput_shared[(((int)threadIdx.x) + 1792)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 609280)];
    PadInput_shared[(((int)threadIdx.x) + 2048)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 802816)];
    PadInput_shared[(((int)threadIdx.x) + 2304)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 809984)];
    PadInput_shared[(((int)threadIdx.x) + 2560)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 1003520)];
    PadInput_shared[(((int)threadIdx.x) + 2816)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 1010688)];
    PadInput_shared[(((int)threadIdx.x) + 3072)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 1204224)];
    PadInput_shared[(((int)threadIdx.x) + 3328)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 1211392)];
    PadInput_shared[(((int)threadIdx.x) + 3584)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 1404928)];
    PadInput_shared[(((int)threadIdx.x) + 3840)] = inputs[((((((((((int)blockIdx.x) / 14) * 14336) + ((((int)threadIdx.x) >> 7) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (rc_0 * 32)) + (((int)threadIdx.x) & 31)) + 1412096)];
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((rc_0 * 2048) + (((int)threadIdx.x) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(weight + (((rc_0 * 2048) + (((int)threadIdx.x) * 4)) + 1024));
    __syncthreads();
    for (int rc_1 = 0; rc_1 < 32; ++rc_1) {
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 128)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 128)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[8] = (conv2d_nhwc_local[8] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 256)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[10] = (conv2d_nhwc_local[10] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 256)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[12] = (conv2d_nhwc_local[12] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 384)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[14] = (conv2d_nhwc_local[14] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 384)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[16] = (conv2d_nhwc_local[16] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2048)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[18] = (conv2d_nhwc_local[18] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2048)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[20] = (conv2d_nhwc_local[20] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2176)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[22] = (conv2d_nhwc_local[22] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2176)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[24] = (conv2d_nhwc_local[24] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2304)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[26] = (conv2d_nhwc_local[26] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2304)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[28] = (conv2d_nhwc_local[28] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2432)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[30] = (conv2d_nhwc_local[30] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2432)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 512)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 512)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 640)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 640)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[9] = (conv2d_nhwc_local[9] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 768)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[11] = (conv2d_nhwc_local[11] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 768)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[13] = (conv2d_nhwc_local[13] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 896)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[15] = (conv2d_nhwc_local[15] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 896)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[17] = (conv2d_nhwc_local[17] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2560)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[19] = (conv2d_nhwc_local[19] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2560)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[21] = (conv2d_nhwc_local[21] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2688)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[23] = (conv2d_nhwc_local[23] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2688)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[25] = (conv2d_nhwc_local[25] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2816)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[27] = (conv2d_nhwc_local[27] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2816)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
      conv2d_nhwc_local[29] = (conv2d_nhwc_local[29] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2944)] * weight_shared[((rc_1 * 64) + (((int)threadIdx.x) & 31))]));
      conv2d_nhwc_local[31] = (conv2d_nhwc_local[31] + (PadInput_shared[(((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + rc_1) + 2944)] * weight_shared[(((rc_1 * 64) + (((int)threadIdx.x) & 31)) + 32)]));
    }
  }
  conv2d_nhwc[((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 3584)] = conv2d_nhwc_local[4];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 3616)] = conv2d_nhwc_local[6];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 7168)] = conv2d_nhwc_local[8];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 7200)] = conv2d_nhwc_local[10];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 10752)] = conv2d_nhwc_local[12];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 10784)] = conv2d_nhwc_local[14];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 802816)] = conv2d_nhwc_local[16];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 802848)] = conv2d_nhwc_local[18];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 806400)] = conv2d_nhwc_local[20];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 806432)] = conv2d_nhwc_local[22];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 809984)] = conv2d_nhwc_local[24];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 810016)] = conv2d_nhwc_local[26];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 813568)] = conv2d_nhwc_local[28];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 813600)] = conv2d_nhwc_local[30];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 200704)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 200736)] = conv2d_nhwc_local[3];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 204288)] = conv2d_nhwc_local[5];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 204320)] = conv2d_nhwc_local[7];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 207872)] = conv2d_nhwc_local[9];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 207904)] = conv2d_nhwc_local[11];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 211456)] = conv2d_nhwc_local[13];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 211488)] = conv2d_nhwc_local[15];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1003520)] = conv2d_nhwc_local[17];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1003552)] = conv2d_nhwc_local[19];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1007104)] = conv2d_nhwc_local[21];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1007136)] = conv2d_nhwc_local[23];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1010688)] = conv2d_nhwc_local[25];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1010720)] = conv2d_nhwc_local[27];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1014272)] = conv2d_nhwc_local[29];
  conv2d_nhwc[(((((((((int)threadIdx.x) >> 7) * 401408) + ((((int)blockIdx.x) / 14) * 14336)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 1014304)] = conv2d_nhwc_local[31];
}

