
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
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[4];
  __shared__ float PadInput_shared[448];
  __shared__ float weight_shared[2048];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 2; ++rc_0) {
    __syncthreads();
    if (((int)threadIdx.x) < 112) {
      *(float4*)(PadInput_shared + (((int)threadIdx.x) * 4)) = *(float4*)(inputs + (((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 4) * 3584)) + ((((int)blockIdx.x) % 28) * 128)) + (((((int)threadIdx.x) & 15) >> 3) * 64)) + (rc_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    }
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((rc_0 * 2048) + (((int)threadIdx.x) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(weight + (((rc_0 * 2048) + (((int)threadIdx.x) * 4)) + 896));
    if (((int)threadIdx.x) < 64) {
      *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(weight + (((rc_0 * 2048) + (((int)threadIdx.x) * 4)) + 1792));
    }
    __syncthreads();
    for (int rc_1 = 0; rc_1 < 4; ++rc_1) {
      for (int w_3 = 0; w_3 < 2; ++w_3) {
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8))] * weight_shared[((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8))] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 1)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 1)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 2)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 128)]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 2)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 129)]));
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 3)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 192)]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 3)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 193)]));
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 4)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 256)]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 4)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 257)]));
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 5)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 320)]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 5)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 321)]));
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 6)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 384)]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 6)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 385)]));
        conv2d_nhwc_local[(w_3 * 2)] = (conv2d_nhwc_local[(w_3 * 2)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 7)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 448)]));
        conv2d_nhwc_local[((w_3 * 2) + 1)] = (conv2d_nhwc_local[((w_3 * 2) + 1)] + (PadInput_shared[(((((((int)threadIdx.x) >> 5) * 64) + (w_3 * 32)) + (rc_1 * 8)) + 7)] * weight_shared[(((rc_1 * 512) + ((((int)threadIdx.x) & 31) * 2)) + 449)]));
      }
    }
  }
  conv2d_nhwc[(((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 28) * 128)) + ((((int)threadIdx.x) & 31) * 2))] = conv2d_nhwc_local[0];
  conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 28) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1)] = conv2d_nhwc_local[1];
  conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 28) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 64)] = conv2d_nhwc_local[2];
  conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + ((((int)blockIdx.x) % 28) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 65)] = conv2d_nhwc_local[3];
}

