
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
  float in_thread_C_shared[1];
  __shared__ float red_result[1];
  __shared__ float C_shared[1];
  in_thread_C_shared[0] = 0.000000e+00f;
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[((int)threadIdx.x)] * A[((int)threadIdx.x)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 512)] * A[(((int)threadIdx.x) + 512)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 1024)] * A[(((int)threadIdx.x) + 1024)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 1536)] * A[(((int)threadIdx.x) + 1536)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 2048)] * A[(((int)threadIdx.x) + 2048)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 2560)] * A[(((int)threadIdx.x) + 2560)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 3072)] * A[(((int)threadIdx.x) + 3072)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 3584)] * A[(((int)threadIdx.x) + 3584)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 4096)] * A[(((int)threadIdx.x) + 4096)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 4608)] * A[(((int)threadIdx.x) + 4608)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 5120)] * A[(((int)threadIdx.x) + 5120)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 5632)] * A[(((int)threadIdx.x) + 5632)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 6144)] * A[(((int)threadIdx.x) + 6144)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 6656)] * A[(((int)threadIdx.x) + 6656)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 7168)] * A[(((int)threadIdx.x) + 7168)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 7680)] * A[(((int)threadIdx.x) + 7680)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 8192)] * A[(((int)threadIdx.x) + 8192)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 8704)] * A[(((int)threadIdx.x) + 8704)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 9216)] * A[(((int)threadIdx.x) + 9216)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 9728)] * A[(((int)threadIdx.x) + 9728)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 10240)] * A[(((int)threadIdx.x) + 10240)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 10752)] * A[(((int)threadIdx.x) + 10752)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 11264)] * A[(((int)threadIdx.x) + 11264)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 11776)] * A[(((int)threadIdx.x) + 11776)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 12288)] * A[(((int)threadIdx.x) + 12288)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 12800)] * A[(((int)threadIdx.x) + 12800)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 13312)] * A[(((int)threadIdx.x) + 13312)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 13824)] * A[(((int)threadIdx.x) + 13824)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 14336)] * A[(((int)threadIdx.x) + 14336)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 14848)] * A[(((int)threadIdx.x) + 14848)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 15360)] * A[(((int)threadIdx.x) + 15360)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 15872)] * A[(((int)threadIdx.x) + 15872)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 16384)] * A[(((int)threadIdx.x) + 16384)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 16896)] * A[(((int)threadIdx.x) + 16896)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 17408)] * A[(((int)threadIdx.x) + 17408)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 17920)] * A[(((int)threadIdx.x) + 17920)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 18432)] * A[(((int)threadIdx.x) + 18432)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 18944)] * A[(((int)threadIdx.x) + 18944)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 19456)] * A[(((int)threadIdx.x) + 19456)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 19968)] * A[(((int)threadIdx.x) + 19968)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 20480)] * A[(((int)threadIdx.x) + 20480)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 20992)] * A[(((int)threadIdx.x) + 20992)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 21504)] * A[(((int)threadIdx.x) + 21504)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 22016)] * A[(((int)threadIdx.x) + 22016)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 22528)] * A[(((int)threadIdx.x) + 22528)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 23040)] * A[(((int)threadIdx.x) + 23040)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 23552)] * A[(((int)threadIdx.x) + 23552)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 24064)] * A[(((int)threadIdx.x) + 24064)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 24576)] * A[(((int)threadIdx.x) + 24576)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 25088)] * A[(((int)threadIdx.x) + 25088)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 25600)] * A[(((int)threadIdx.x) + 25600)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 26112)] * A[(((int)threadIdx.x) + 26112)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 26624)] * A[(((int)threadIdx.x) + 26624)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 27136)] * A[(((int)threadIdx.x) + 27136)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 27648)] * A[(((int)threadIdx.x) + 27648)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 28160)] * A[(((int)threadIdx.x) + 28160)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 28672)] * A[(((int)threadIdx.x) + 28672)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 29184)] * A[(((int)threadIdx.x) + 29184)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 29696)] * A[(((int)threadIdx.x) + 29696)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 30208)] * A[(((int)threadIdx.x) + 30208)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 30720)] * A[(((int)threadIdx.x) + 30720)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 31232)] * A[(((int)threadIdx.x) + 31232)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 31744)] * A[(((int)threadIdx.x) + 31744)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 32256)] * A[(((int)threadIdx.x) + 32256)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 32768)] * A[(((int)threadIdx.x) + 32768)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 33280)] * A[(((int)threadIdx.x) + 33280)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 33792)] * A[(((int)threadIdx.x) + 33792)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 34304)] * A[(((int)threadIdx.x) + 34304)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 34816)] * A[(((int)threadIdx.x) + 34816)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 35328)] * A[(((int)threadIdx.x) + 35328)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 35840)] * A[(((int)threadIdx.x) + 35840)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 36352)] * A[(((int)threadIdx.x) + 36352)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 36864)] * A[(((int)threadIdx.x) + 36864)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 37376)] * A[(((int)threadIdx.x) + 37376)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 37888)] * A[(((int)threadIdx.x) + 37888)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 38400)] * A[(((int)threadIdx.x) + 38400)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 38912)] * A[(((int)threadIdx.x) + 38912)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 39424)] * A[(((int)threadIdx.x) + 39424)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 39936)] * A[(((int)threadIdx.x) + 39936)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 40448)] * A[(((int)threadIdx.x) + 40448)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 40960)] * A[(((int)threadIdx.x) + 40960)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 41472)] * A[(((int)threadIdx.x) + 41472)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 41984)] * A[(((int)threadIdx.x) + 41984)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 42496)] * A[(((int)threadIdx.x) + 42496)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 43008)] * A[(((int)threadIdx.x) + 43008)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 43520)] * A[(((int)threadIdx.x) + 43520)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 44032)] * A[(((int)threadIdx.x) + 44032)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 44544)] * A[(((int)threadIdx.x) + 44544)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 45056)] * A[(((int)threadIdx.x) + 45056)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 45568)] * A[(((int)threadIdx.x) + 45568)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 46080)] * A[(((int)threadIdx.x) + 46080)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 46592)] * A[(((int)threadIdx.x) + 46592)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 47104)] * A[(((int)threadIdx.x) + 47104)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 47616)] * A[(((int)threadIdx.x) + 47616)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 48128)] * A[(((int)threadIdx.x) + 48128)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 48640)] * A[(((int)threadIdx.x) + 48640)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 49152)] * A[(((int)threadIdx.x) + 49152)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 49664)] * A[(((int)threadIdx.x) + 49664)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 50176)] * A[(((int)threadIdx.x) + 50176)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 50688)] * A[(((int)threadIdx.x) + 50688)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 51200)] * A[(((int)threadIdx.x) + 51200)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 51712)] * A[(((int)threadIdx.x) + 51712)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 52224)] * A[(((int)threadIdx.x) + 52224)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 52736)] * A[(((int)threadIdx.x) + 52736)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 53248)] * A[(((int)threadIdx.x) + 53248)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 53760)] * A[(((int)threadIdx.x) + 53760)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 54272)] * A[(((int)threadIdx.x) + 54272)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 54784)] * A[(((int)threadIdx.x) + 54784)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 55296)] * A[(((int)threadIdx.x) + 55296)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 55808)] * A[(((int)threadIdx.x) + 55808)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 56320)] * A[(((int)threadIdx.x) + 56320)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 56832)] * A[(((int)threadIdx.x) + 56832)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 57344)] * A[(((int)threadIdx.x) + 57344)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 57856)] * A[(((int)threadIdx.x) + 57856)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 58368)] * A[(((int)threadIdx.x) + 58368)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 58880)] * A[(((int)threadIdx.x) + 58880)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 59392)] * A[(((int)threadIdx.x) + 59392)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 59904)] * A[(((int)threadIdx.x) + 59904)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 60416)] * A[(((int)threadIdx.x) + 60416)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 60928)] * A[(((int)threadIdx.x) + 60928)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 61440)] * A[(((int)threadIdx.x) + 61440)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 61952)] * A[(((int)threadIdx.x) + 61952)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 62464)] * A[(((int)threadIdx.x) + 62464)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 62976)] * A[(((int)threadIdx.x) + 62976)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 63488)] * A[(((int)threadIdx.x) + 63488)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 64000)] * A[(((int)threadIdx.x) + 64000)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 64512)] * A[(((int)threadIdx.x) + 64512)]));
  in_thread_C_shared[0] = (in_thread_C_shared[0] + (A[(((int)threadIdx.x) + 65024)] * A[(((int)threadIdx.x) + 65024)]));
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
    C_shared[0] = ((volatile float*)red_result)[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 1) {
    D[0] = sqrtf(C_shared[0]);
  }
}

