
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


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
extern "C" __global__ void __launch_bounds__(512) main_kernel(float* __restrict__ A, float* __restrict__ D);
extern "C" __global__ void __launch_bounds__(512) main_kernel(float* __restrict__ A, float* __restrict__ D) {
  __shared__ float C_shared[8];
  float in_thread_C_shared[1];
  __shared__ float red_result[1];
  for (int ax0 = 0; ax0 < 8; ++ax0) {
    in_thread_C_shared[0] = 0.000000e+00f;
    for (int ax1_ax2_fused_0 = 0; ax1_ax2_fused_0 < 128; ++ax1_ax2_fused_0) {
      in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((ax0 * 65536) + (ax1_ax2_fused_0 * 512)) + ((int)threadIdx.x))] * A[(((ax0 * 65536) + (ax1_ax2_fused_0 * 512)) + ((int)threadIdx.x))]));
    }
    float red_buf0[1];
    uint mask[1];
    float t0[1];
    float red_buf0_1[1];
    uint mask_1[1];
    float t0_1[1];
    __shared__ float red_buf_staging[16];
    red_buf0_1[0] = in_thread_C_shared[0];
    mask_1[0] = __activemask();
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
    red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
    red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
    red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
    red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
    red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
    if ((((int)threadIdx.x) % 32) == 0) {
      red_buf_staging[(((int)threadIdx.x) >> 5)] = red_buf0_1[0];
    }
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      red_buf0[0] = red_buf_staging[((int)threadIdx.x)];
    }
    mask[0] = __activemask();
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    if (((int)threadIdx.x) == 0) {
      ((volatile float*)red_result)[0] = red_buf0[0];
    }
    __syncthreads();
    if (((int)threadIdx.x) == 0) {
      C_shared[ax0] = ((volatile float*)red_result)[0];
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    D[((int)threadIdx.x)] = sqrtf(C_shared[((int)threadIdx.x)]);
  }
}

