
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
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A, float* __restrict__ T_softmax_norm);
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A, float* __restrict__ T_softmax_norm) {
  float in_thread_T_softmax_maxelem_shared[1];
  float red_buf0[1];
  __shared__ float T_softmax_maxelem_shared[1];
  float in_thread_T_softmax_expsum_shared[1];
  float red_buf0_1[1];
  __shared__ float T_softmax_expsum_shared[1];
  in_thread_T_softmax_maxelem_shared[0] = -3.402823e+38f;
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 16)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 32)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 48)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 64)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 80)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 96)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 112)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 128)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 144)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 160)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 176)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 192)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 208)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 224)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 240)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 272)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 288)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 304)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 320)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 336)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 352)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 368)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 384)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 400)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 416)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 432)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 448)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 464)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 480)]);
  in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 496)]);
  uint mask[1];
  float t0[1];
  red_buf0[0] = in_thread_T_softmax_maxelem_shared[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  if (((int)threadIdx.x) == 0) {
    T_softmax_maxelem_shared[0] = red_buf0[0];
  }
  in_thread_T_softmax_expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 16)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 32)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 48)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 64)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 80)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 96)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 112)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 128)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 144)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 160)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 176)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 192)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 208)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 224)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 240)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 272)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 288)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 304)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 320)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 336)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 352)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 368)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 384)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 400)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 416)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 432)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 448)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 464)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 480)] - T_softmax_maxelem_shared[0])));
  in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 496)] - T_softmax_maxelem_shared[0])));
  uint mask_1[1];
  float t0_1[1];
  red_buf0_1[0] = in_thread_T_softmax_expsum_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf0_1[0] = __shfl_sync(mask_1[0], red_buf0_1[0], 0, 32);
  if (((int)threadIdx.x) == 0) {
    T_softmax_expsum_shared[0] = red_buf0_1[0];
  }
  __syncthreads();
  T_softmax_norm[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = (__expf((A[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 16)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 16)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 32)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 32)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 48)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 48)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 64)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 64)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 80)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 80)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 96)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 96)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 112)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 112)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 128)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 128)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 144)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 144)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 160)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 160)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 176)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 176)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 192)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 192)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 208)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 208)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 224)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 224)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 240)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 240)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 272)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 272)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 288)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 288)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 304)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 304)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 320)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 320)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 336)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 336)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 352)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 352)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 368)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 368)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 384)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 384)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 400)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 400)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 416)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 416)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 432)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 432)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 448)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 448)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 464)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 464)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 480)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 480)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
  T_softmax_norm[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 496)] = (__expf((A[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 496)] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]);
}

