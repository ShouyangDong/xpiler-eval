
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
extern "C" __global__ void __launch_bounds__(112) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(112) main_kernel(float* __restrict__ depth_conv2d_nhwc, float* __restrict__ placeholder, float* __restrict__ placeholder_1) {
  float depth_conv2d_nhwc_local[7];
  __shared__ float PadInput_shared[3024];
  __shared__ float placeholder_shared[144];
  depth_conv2d_nhwc_local[0] = 0.000000e+00f;
  depth_conv2d_nhwc_local[1] = 0.000000e+00f;
  depth_conv2d_nhwc_local[2] = 0.000000e+00f;
  depth_conv2d_nhwc_local[3] = 0.000000e+00f;
  depth_conv2d_nhwc_local[4] = 0.000000e+00f;
  depth_conv2d_nhwc_local[5] = 0.000000e+00f;
  depth_conv2d_nhwc_local[6] = 0.000000e+00f;
  float4 condval;
  if (((28 <= ((int)threadIdx.x)) && (4 <= (((int)threadIdx.x) % 28)))) {
    condval = *(float4*)(placeholder + (((((((int)threadIdx.x) >> 2) * 1024) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 8192));
  } else {
    condval = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  *(float4*)(PadInput_shared + (((int)threadIdx.x) * 4)) = condval;
  float4 condval_1;
  if ((4 <= (((int)threadIdx.x) % 28))) {
    condval_1 = *(float4*)(placeholder + (((((((int)threadIdx.x) >> 2) * 1024) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 20480));
  } else {
    condval_1 = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 448)) = condval_1;
  if (((int)threadIdx.x) < 28) {
    *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 896)) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 12) {
    *(float4*)(placeholder_shared + (((int)threadIdx.x) * 4)) = *(float4*)(placeholder_1 + ((((((int)threadIdx.x) >> 2) * 3072) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float4 condval_2;
  if ((28 <= ((int)threadIdx.x))) {
    condval_2 = *(float4*)(placeholder + (((((((int)threadIdx.x) >> 2) * 1024) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 7168));
  } else {
    condval_2 = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 1008)) = condval_2;
  *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 1456)) = *(float4*)(placeholder + (((((((int)threadIdx.x) >> 2) * 1024) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 21504));
  if (((int)threadIdx.x) < 28) {
    *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 1904)) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 12) {
    *(float4*)(placeholder_shared + ((((int)threadIdx.x) * 4) + 48)) = *(float4*)(placeholder_1 + (((((((int)threadIdx.x) >> 2) * 3072) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1024));
  }
__asm__ __volatile__("cp.async.commit_group;");

  float4 condval_3;
  if (((28 <= ((int)threadIdx.x)) && ((((int)threadIdx.x) % 28) < 24))) {
    condval_3 = *(float4*)(placeholder + (((((((int)threadIdx.x) >> 2) * 1024) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) - 6144));
  } else {
    condval_3 = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 2016)) = condval_3;
  float4 condval_4;
  if (((((int)threadIdx.x) % 28) < 24)) {
    condval_4 = *(float4*)(placeholder + (((((((int)threadIdx.x) >> 2) * 1024) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 22528));
  } else {
    condval_4 = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 2464)) = condval_4;
  if (((int)threadIdx.x) < 28) {
    *(float4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 2912)) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 12) {
    *(float4*)(placeholder_shared + ((((int)threadIdx.x) * 4) + 96)) = *(float4*)(placeholder_1 + (((((((int)threadIdx.x) >> 2) * 3072) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2048));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((int)threadIdx.x)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 112)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 224)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 336)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 448)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 560)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 672)] * placeholder_shared[(((int)threadIdx.x) & 15)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 112)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 224)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 336)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 448)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 560)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 672)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 784)] * placeholder_shared[((((int)threadIdx.x) & 15) + 16)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 224)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 336)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 448)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 560)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 672)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 784)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 896)] * placeholder_shared[((((int)threadIdx.x) & 15) + 32)]));
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 1008)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 1232)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1456)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1568)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 1680)] * placeholder_shared[((((int)threadIdx.x) & 15) + 48)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 1120)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 1232)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 1456)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1568)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1680)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 1792)] * placeholder_shared[((((int)threadIdx.x) & 15) + 64)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 1232)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 1344)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 1456)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 1568)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 1680)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 1792)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 1904)] * placeholder_shared[((((int)threadIdx.x) & 15) + 80)]));
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 2016)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 2128)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 2352)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 2464)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 2576)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 15) + 96)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 2128)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 2352)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 2464)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 2576)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2800)] * placeholder_shared[((((int)threadIdx.x) & 15) + 112)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) + 2240)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[(((int)threadIdx.x) + 2352)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[(((int)threadIdx.x) + 2464)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[(((int)threadIdx.x) + 2576)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[(((int)threadIdx.x) + 2688)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[(((int)threadIdx.x) + 2800)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[(((int)threadIdx.x) + 2912)] * placeholder_shared[((((int)threadIdx.x) & 15) + 128)]));
  depth_conv2d_nhwc[((((((int)threadIdx.x) >> 4) * 1024) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[(((((((int)threadIdx.x) >> 4) * 1024) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15)) + 7168)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[(((((((int)threadIdx.x) >> 4) * 1024) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15)) + 14336)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[(((((((int)threadIdx.x) >> 4) * 1024) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15)) + 21504)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[(((((((int)threadIdx.x) >> 4) * 1024) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15)) + 28672)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[(((((((int)threadIdx.x) >> 4) * 1024) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15)) + 35840)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[(((((((int)threadIdx.x) >> 4) * 1024) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15)) + 43008)] = depth_conv2d_nhwc_local[6];
}

