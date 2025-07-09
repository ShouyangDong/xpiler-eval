
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
  __shared__ float PadInput_shared[256];
  __shared__ float weight_shared[2048];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[8] = 0.000000e+00f;
  conv2d_nhwc_local[9] = 0.000000e+00f;
  conv2d_nhwc_local[12] = 0.000000e+00f;
  conv2d_nhwc_local[13] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  conv2d_nhwc_local[10] = 0.000000e+00f;
  conv2d_nhwc_local[11] = 0.000000e+00f;
  conv2d_nhwc_local[14] = 0.000000e+00f;
  conv2d_nhwc_local[15] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 2; ++rc_0) {
    __syncthreads();
    PadInput_shared[((int)threadIdx.x)] = inputs[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x))];
    PadInput_shared[(((int)threadIdx.x) + 32)] = inputs[((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x)) + 64)];
    PadInput_shared[(((int)threadIdx.x) + 64)] = inputs[((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x)) + 3584)];
    PadInput_shared[(((int)threadIdx.x) + 96)] = inputs[((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x)) + 3648)];
    PadInput_shared[(((int)threadIdx.x) + 128)] = inputs[((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x)) + 7168)];
    PadInput_shared[(((int)threadIdx.x) + 160)] = inputs[((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x)) + 7232)];
    PadInput_shared[(((int)threadIdx.x) + 192)] = inputs[((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x)) + 10752)];
    PadInput_shared[(((int)threadIdx.x) + 224)] = inputs[((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (rc_0 * 32)) + ((int)threadIdx.x)) + 10816)];
    for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 64; ++ax0_ax1_ax2_ax3_fused_0) {
      weight_shared[((ax0_ax1_ax2_ax3_fused_0 * 32) + ((int)threadIdx.x))] = weight[(((rc_0 * 2048) + (ax0_ax1_ax2_ax3_fused_0 * 32)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int rc_1 = 0; rc_1 < 8; ++rc_1) {
      for (int w_3 = 0; w_3 < 2; ++w_3) {
        for (int rc_2 = 0; rc_2 < 4; ++rc_2) {
          conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((w_3 * 32) + (rc_1 * 4)) + rc_2)] * weight_shared[(((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2))]));
          conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((w_3 * 32) + (rc_1 * 4)) + rc_2)] * weight_shared[((((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2)) + 1)]));
          conv2d_nhwc_local[((w_3 * 2) + 4)] = (conv2d_nhwc_local[((w_3 * 2) + 4)] + (PadInput_shared[((((w_3 * 32) + (rc_1 * 4)) + rc_2) + 64)] * weight_shared[(((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2))]));
          conv2d_nhwc_local[((w_3 * 2) + 5)] = (conv2d_nhwc_local[((w_3 * 2) + 5)] + (PadInput_shared[((((w_3 * 32) + (rc_1 * 4)) + rc_2) + 64)] * weight_shared[((((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2)) + 1)]));
          conv2d_nhwc_local[((w_3 * 2) + 8)] = (conv2d_nhwc_local[((w_3 * 2) + 8)] + (PadInput_shared[((((w_3 * 32) + (rc_1 * 4)) + rc_2) + 128)] * weight_shared[(((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2))]));
          conv2d_nhwc_local[((w_3 * 2) + 9)] = (conv2d_nhwc_local[((w_3 * 2) + 9)] + (PadInput_shared[((((w_3 * 32) + (rc_1 * 4)) + rc_2) + 128)] * weight_shared[((((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2)) + 1)]));
          conv2d_nhwc_local[((w_3 * 2) + 12)] = (conv2d_nhwc_local[((w_3 * 2) + 12)] + (PadInput_shared[((((w_3 * 32) + (rc_1 * 4)) + rc_2) + 192)] * weight_shared[(((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2))]));
          conv2d_nhwc_local[((w_3 * 2) + 13)] = (conv2d_nhwc_local[((w_3 * 2) + 13)] + (PadInput_shared[((((w_3 * 32) + (rc_1 * 4)) + rc_2) + 192)] * weight_shared[((((rc_1 * 256) + (rc_2 * 64)) + (((int)threadIdx.x) * 2)) + 1)]));
        }
      }
    }
  }
  conv2d_nhwc[((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 1)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 64)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 65)] = conv2d_nhwc_local[3];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 3584)] = conv2d_nhwc_local[4];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 3585)] = conv2d_nhwc_local[5];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 3648)] = conv2d_nhwc_local[6];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 3649)] = conv2d_nhwc_local[7];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 7168)] = conv2d_nhwc_local[8];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 7169)] = conv2d_nhwc_local[9];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 7232)] = conv2d_nhwc_local[10];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 7233)] = conv2d_nhwc_local[11];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 10752)] = conv2d_nhwc_local[12];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 10753)] = conv2d_nhwc_local[13];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 10816)] = conv2d_nhwc_local[14];
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 128)) + (((int)threadIdx.x) * 2)) + 10817)] = conv2d_nhwc_local[15];
}

