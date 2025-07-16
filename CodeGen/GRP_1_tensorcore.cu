
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
  float conv2d_nhwc_local[16];
  __shared__ float PadInput_shared[512];
  __shared__ float weight_shared[512];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[8] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[9] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[10] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[11] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[12] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[13] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[14] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  conv2d_nhwc_local[15] = 0.000000e+00f;
  PadInput_shared[((int)threadIdx.x)] = inputs[(((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x))];
  PadInput_shared[(((int)threadIdx.x) + 32)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 64)];
  PadInput_shared[(((int)threadIdx.x) + 64)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3584)];
  PadInput_shared[(((int)threadIdx.x) + 96)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3648)];
  PadInput_shared[(((int)threadIdx.x) + 128)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7168)];
  PadInput_shared[(((int)threadIdx.x) + 160)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7232)];
  PadInput_shared[(((int)threadIdx.x) + 192)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10752)];
  PadInput_shared[(((int)threadIdx.x) + 224)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10816)];
  PadInput_shared[(((int)threadIdx.x) + 256)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14336)];
  PadInput_shared[(((int)threadIdx.x) + 288)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14400)];
  PadInput_shared[(((int)threadIdx.x) + 320)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 17920)];
  PadInput_shared[(((int)threadIdx.x) + 352)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 17984)];
  PadInput_shared[(((int)threadIdx.x) + 384)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21504)];
  PadInput_shared[(((int)threadIdx.x) + 416)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21568)];
  PadInput_shared[(((int)threadIdx.x) + 448)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 25088)];
  PadInput_shared[(((int)threadIdx.x) + 480)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 25152)];
  weight_shared[((int)threadIdx.x)] = weight[(((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x))];
  weight_shared[(((int)threadIdx.x) + 32)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 64)];
  weight_shared[(((int)threadIdx.x) + 64)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 128)];
  weight_shared[(((int)threadIdx.x) + 96)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 192)];
  weight_shared[(((int)threadIdx.x) + 128)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 256)];
  weight_shared[(((int)threadIdx.x) + 160)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 320)];
  weight_shared[(((int)threadIdx.x) + 192)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 384)];
  weight_shared[(((int)threadIdx.x) + 224)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 448)];
  weight_shared[(((int)threadIdx.x) + 256)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 512)];
  weight_shared[(((int)threadIdx.x) + 288)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 576)];
  weight_shared[(((int)threadIdx.x) + 320)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 640)];
  weight_shared[(((int)threadIdx.x) + 352)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 704)];
  weight_shared[(((int)threadIdx.x) + 384)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 768)];
  weight_shared[(((int)threadIdx.x) + 416)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 832)];
  weight_shared[(((int)threadIdx.x) + 448)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 896)];
  weight_shared[(((int)threadIdx.x) + 480)] = weight[((((((int)blockIdx.x) & 1) * 32) + ((int)threadIdx.x)) + 960)];
  __syncthreads();
  for (int rc_2 = 0; rc_2 < 16; ++rc_2) {
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 16) + rc_2)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[8] = (conv2d_nhwc_local[8] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 32)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 64)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[9] = (conv2d_nhwc_local[9] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 96)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 128)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[10] = (conv2d_nhwc_local[10] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 160)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 192)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[11] = (conv2d_nhwc_local[11] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 224)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 256)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[12] = (conv2d_nhwc_local[12] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 288)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 320)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[13] = (conv2d_nhwc_local[13] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 352)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 384)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[14] = (conv2d_nhwc_local[14] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 416)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 448)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
    conv2d_nhwc_local[15] = (conv2d_nhwc_local[15] + (PadInput_shared[((((((int)threadIdx.x) >> 4) * 16) + rc_2) + 480)] * weight_shared[((rc_2 * 32) + ((int)threadIdx.x))]));
  }
  conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x))] = conv2d_nhwc_local[0];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 64)] = conv2d_nhwc_local[8];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3584)] = conv2d_nhwc_local[1];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 3648)] = conv2d_nhwc_local[9];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7168)] = conv2d_nhwc_local[2];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 7232)] = conv2d_nhwc_local[10];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10752)] = conv2d_nhwc_local[3];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 10816)] = conv2d_nhwc_local[11];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14336)] = conv2d_nhwc_local[4];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 14400)] = conv2d_nhwc_local[12];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 17920)] = conv2d_nhwc_local[5];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 17984)] = conv2d_nhwc_local[13];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21504)] = conv2d_nhwc_local[6];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 21568)] = conv2d_nhwc_local[14];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 25088)] = conv2d_nhwc_local[7];
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 1) * 128)) + ((((int)blockIdx.x) & 1) * 32)) + ((int)threadIdx.x)) + 25152)] = conv2d_nhwc_local[15];
}

