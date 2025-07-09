
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
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[8];
  __shared__ float PadInput_shared[1024];
  __shared__ float weight_shared[512];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  PadInput_shared[((int)threadIdx.x)] = inputs[((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31))];
  PadInput_shared[(((int)threadIdx.x) + 128)] = inputs[(((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
  PadInput_shared[(((int)threadIdx.x) + 256)] = inputs[(((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
  PadInput_shared[(((int)threadIdx.x) + 384)] = inputs[(((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 10752)];
  PadInput_shared[(((int)threadIdx.x) + 512)] = inputs[(((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
  PadInput_shared[(((int)threadIdx.x) + 640)] = inputs[(((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 17920)];
  PadInput_shared[(((int)threadIdx.x) + 768)] = inputs[(((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 21504)];
  PadInput_shared[(((int)threadIdx.x) + 896)] = inputs[(((((((((int)blockIdx.x) / 28) * 28672) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 25088)];
  weight_shared[((int)threadIdx.x)] = weight[((((((int)threadIdx.x) >> 5) * 64) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31))];
  weight_shared[(((int)threadIdx.x) + 128)] = weight[(((((((int)threadIdx.x) >> 5) * 64) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 256)];
  weight_shared[(((int)threadIdx.x) + 256)] = weight[(((((((int)threadIdx.x) >> 5) * 64) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 512)];
  weight_shared[(((int)threadIdx.x) + 384)] = weight[(((((((int)threadIdx.x) >> 5) * 64) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)) + 768)];
  __syncthreads();
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64))] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 16)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 32)] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 48)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 128)] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 144)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 160)] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 176)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 1)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 17)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 33)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 49)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 129)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 145)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 161)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 177)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 2)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 18)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 34)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 50)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 130)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 146)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 162)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 178)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 3)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 19)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 35)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 51)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 131)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 147)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 163)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 179)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 4)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 20)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 36)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 52)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 132)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 148)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 164)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 180)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 5)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 21)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 37)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 53)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 133)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 149)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 165)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 181)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 6)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 22)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 38)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 54)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 134)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 150)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 166)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 182)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 7)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 23)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 39)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 55)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 135)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 151)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 167)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 183)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 8)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 24)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 40)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 56)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 136)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 152)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 168)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 184)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 9)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 25)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 41)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 57)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 137)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 153)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 169)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 185)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 10)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 26)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 42)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 58)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 138)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 154)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 170)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 186)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 11)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 27)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 43)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 59)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 139)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 155)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 171)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 187)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 12)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 28)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 44)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 60)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 140)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 156)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 172)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 188)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 13)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 29)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 45)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 61)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 141)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 157)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 173)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 189)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 14)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 30)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 46)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 62)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 142)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 158)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 174)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 190)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 15)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 31)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 47)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 63)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 143)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 159)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 175)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + 191)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc[(((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15))] = conv2d_nhwc_local[0];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = conv2d_nhwc_local[4];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 64)] = conv2d_nhwc_local[1];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 80)] = conv2d_nhwc_local[5];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 3584)] = conv2d_nhwc_local[2];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 3600)] = conv2d_nhwc_local[6];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 3648)] = conv2d_nhwc_local[3];
  conv2d_nhwc[((((((((((int)blockIdx.x) / 28) * 28672) + ((((int)threadIdx.x) >> 5) * 7168)) + (((((int)blockIdx.x) % 28) >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 3664)] = conv2d_nhwc_local[7];
}

