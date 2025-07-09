
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
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float Conv2dOutput_local[4];
  __shared__ float PaddedInput_shared[5184];
  __shared__ float kernel_shared[2048];
  Conv2dOutput_local[0] = 0.000000e+00f;
  Conv2dOutput_local[2] = 0.000000e+00f;
  Conv2dOutput_local[1] = 0.000000e+00f;
  Conv2dOutput_local[3] = 0.000000e+00f;
  *(float4*)(PaddedInput_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + (((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(data + ((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + (((int)threadIdx.x) * 4)) = *(float4*)(kernel + ((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
  }
__asm__ __volatile__("cp.async.commit_group;");

  *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 1296)) = *(float4*)(data + ((((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 16));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 2192)) = *(float4*)(data + (((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 16));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(kernel + (((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 4096));
  }
__asm__ __volatile__("cp.async.commit_group;");

  *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 2592)) = *(float4*)(data + ((((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 32));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 3488)) = *(float4*)(data + (((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 32));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(kernel + (((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 8192));
  }
__asm__ __volatile__("cp.async.commit_group;");

  *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 3888)) = *(float4*)(data + ((((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 48));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 4784)) = *(float4*)(data + (((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 48));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(kernel + (((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 12288));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[((((int)threadIdx.x) >> 4) * 96)] * kernel_shared[((((int)threadIdx.x) & 15) * 2)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 32)] * kernel_shared[((((int)threadIdx.x) & 15) * 2)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 33)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 34)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 35)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 4)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 36)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 5)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 37)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 6)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 38)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 7)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 39)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 8)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 40)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 9)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 41)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 10)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 42)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 11)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 43)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 12)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 44)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 13)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 45)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 14)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 46)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 15)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 47)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[((((int)threadIdx.x) >> 4) * 96)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 32)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 33)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 34)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 35)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 4)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 36)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 5)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 37)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 6)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 38)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 7)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 39)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 8)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 40)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 9)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 41)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 10)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 42)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 11)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 43)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 12)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 44)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 13)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 45)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 14)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 46)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 15)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 47)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
  __syncthreads();
  *(float4*)(PaddedInput_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + ((((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 64));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(data + (((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 64));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + (((int)threadIdx.x) * 4)) = *(float4*)(kernel + (((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 16384));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1296)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1328)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1297)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1329)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1298)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1330)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1299)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1331)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1300)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1332)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1301)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1333)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1302)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1334)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1303)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1335)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1304)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 768)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1336)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 768)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1305)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 800)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1337)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 800)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1306)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 832)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1338)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 832)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1307)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 864)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1339)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 864)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1308)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 896)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1340)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 896)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1309)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 928)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1341)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 928)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1310)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 960)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1342)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 960)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1311)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 992)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1343)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 992)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1296)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1328)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1297)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1329)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1298)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1330)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1299)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1331)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1300)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1332)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1301)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1333)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1302)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1334)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1303)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1335)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1304)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 769)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1336)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 769)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1305)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 801)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1337)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 801)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1306)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 833)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1338)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 833)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1307)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 865)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1339)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 865)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1308)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 897)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1340)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 897)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1309)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 929)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1341)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 929)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1310)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 961)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1342)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 961)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1311)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 993)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1343)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 993)]));
  __syncthreads();
  *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 1296)) = *(float4*)(data + ((((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 80));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 2192)) = *(float4*)(data + (((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 80));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(kernel + (((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 20480));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2592)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1024)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2624)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1024)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2593)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1056)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2625)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1056)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2594)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1088)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2626)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1088)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2595)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1120)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2627)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1120)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2596)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1152)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2628)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1152)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2597)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1184)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2629)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1184)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2598)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1216)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2630)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1216)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2599)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1248)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2631)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1248)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2600)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1280)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2632)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1280)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2601)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1312)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2633)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1312)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2602)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1344)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2634)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1344)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2603)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1376)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2635)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1376)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2604)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1408)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2636)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1408)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2605)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1440)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2637)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1440)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2606)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1472)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2638)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1472)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2607)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1504)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2639)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1504)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2592)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1025)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2624)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1025)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2593)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1057)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2625)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1057)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2594)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1089)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2626)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1089)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2595)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1121)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2627)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1121)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2596)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1153)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2628)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1153)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2597)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1185)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2629)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1185)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2598)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1217)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2630)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1217)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2599)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1249)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2631)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1249)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2600)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1281)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2632)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1281)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2601)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1313)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2633)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1313)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2602)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1345)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2634)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1345)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2603)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1377)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2635)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1377)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2604)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1409)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2636)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1409)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2605)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1441)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2637)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1441)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2606)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1473)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2638)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1473)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2607)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1505)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2639)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1505)]));
  __syncthreads();
  *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 2592)) = *(float4*)(data + ((((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 96));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 3488)) = *(float4*)(data + (((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 96));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(kernel + (((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 24576));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3888)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1536)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3920)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1536)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3889)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1568)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3921)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1568)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3890)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1600)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3922)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1600)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3891)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1632)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3923)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1632)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3892)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1664)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3924)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1664)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3893)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1696)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3925)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1696)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3894)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1728)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3926)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1728)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3895)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1760)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3927)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1760)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3896)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1792)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3928)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1792)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3897)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1824)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3929)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1824)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3898)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1856)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3930)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1856)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3899)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1888)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3931)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1888)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3900)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1920)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3932)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1920)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3901)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1952)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3933)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1952)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3902)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1984)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3934)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1984)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3903)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2016)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3935)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2016)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3888)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1537)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3920)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1537)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3889)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1569)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3921)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1569)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3890)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1601)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3922)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1601)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3891)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1633)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3923)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1633)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3892)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1665)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3924)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1665)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3893)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1697)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3925)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1697)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3894)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1729)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3926)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1729)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3895)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1761)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3927)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1761)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3896)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1793)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3928)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1793)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3897)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1825)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3929)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1825)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3898)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1857)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3930)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1857)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3899)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1889)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3931)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1889)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3900)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1921)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3932)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1921)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3901)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1953)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3933)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1953)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3902)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1985)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3934)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1985)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3903)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2017)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3935)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2017)]));
  __syncthreads();
  *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 3888)) = *(float4*)(data + ((((((((int)threadIdx.x) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + (((((int)threadIdx.x) % 12) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 112));
  if (((int)threadIdx.x) < 100) {
    *(float4*)(PaddedInput_shared + ((((int)threadIdx.x) * 4) + 4784)) = *(float4*)(data + (((((((((int)threadIdx.x) + 224) / 12) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((((int)threadIdx.x) >> 2) + 2) % 3) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 112));
  }
  if (((int)threadIdx.x) < 128) {
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(kernel + (((((((int)threadIdx.x) >> 3) * 256) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 28672));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[((((int)threadIdx.x) >> 4) * 96)] * kernel_shared[((((int)threadIdx.x) & 15) * 2)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 32)] * kernel_shared[((((int)threadIdx.x) & 15) * 2)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 33)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 32)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 34)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 64)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 35)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 96)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 4)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 36)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 128)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 5)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 37)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 160)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 6)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 38)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 192)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 7)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 39)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 224)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 8)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 40)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 256)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 9)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 41)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 288)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 10)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 42)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 320)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 11)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 43)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 352)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 12)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 44)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 384)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 13)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 45)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 416)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 14)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 46)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 448)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 15)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 47)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 480)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[((((int)threadIdx.x) >> 4) * 96)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 32)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 33)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 33)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 34)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 65)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 35)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 97)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 4)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 36)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 129)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 5)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 37)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 161)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 6)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 38)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 193)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 7)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 39)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 225)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 8)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 40)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 257)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 9)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 41)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 289)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 10)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 42)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 321)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 11)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 43)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 353)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 12)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 44)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 385)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 13)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 45)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 417)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 14)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 46)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 449)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 15)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 47)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 481)]));
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1296)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1328)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 512)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1297)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1329)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 544)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1298)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1330)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 576)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1299)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1331)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 608)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1300)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1332)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 640)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1301)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1333)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 672)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1302)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1334)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 704)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1303)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1335)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 736)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1304)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 768)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1336)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 768)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1305)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 800)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1337)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 800)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1306)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 832)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1338)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 832)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1307)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 864)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1339)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 864)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1308)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 896)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1340)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 896)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1309)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 928)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1341)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 928)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1310)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 960)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1342)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 960)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1311)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 992)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1343)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 992)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1296)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1328)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 513)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1297)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1329)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 545)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1298)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1330)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 577)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1299)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1331)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 609)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1300)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1332)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 641)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1301)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1333)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 673)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1302)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1334)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 705)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1303)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1335)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 737)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1304)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 769)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1336)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 769)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1305)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 801)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1337)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 801)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1306)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 833)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1338)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 833)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1307)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 865)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1339)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 865)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1308)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 897)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1340)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 897)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1309)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 929)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1341)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 929)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1310)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 961)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1342)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 961)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1311)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 993)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 1343)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 993)]));
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2592)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1024)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2624)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1024)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2593)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1056)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2625)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1056)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2594)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1088)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2626)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1088)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2595)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1120)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2627)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1120)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2596)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1152)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2628)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1152)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2597)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1184)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2629)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1184)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2598)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1216)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2630)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1216)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2599)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1248)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2631)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1248)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2600)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1280)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2632)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1280)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2601)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1312)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2633)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1312)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2602)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1344)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2634)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1344)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2603)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1376)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2635)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1376)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2604)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1408)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2636)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1408)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2605)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1440)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2637)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1440)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2606)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1472)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2638)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1472)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2607)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1504)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2639)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1504)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2592)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1025)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2624)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1025)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2593)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1057)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2625)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1057)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2594)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1089)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2626)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1089)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2595)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1121)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2627)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1121)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2596)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1153)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2628)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1153)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2597)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1185)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2629)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1185)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2598)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1217)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2630)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1217)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2599)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1249)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2631)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1249)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2600)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1281)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2632)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1281)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2601)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1313)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2633)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1313)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2602)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1345)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2634)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1345)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2603)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1377)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2635)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1377)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2604)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1409)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2636)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1409)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2605)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1441)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2637)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1441)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2606)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1473)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2638)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1473)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2607)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1505)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 2639)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1505)]));
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3888)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1536)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3920)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1536)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3889)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1568)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3921)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1568)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3890)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1600)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3922)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1600)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3891)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1632)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3923)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1632)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3892)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1664)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3924)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1664)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3893)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1696)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3925)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1696)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3894)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1728)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3926)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1728)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3895)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1760)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3927)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1760)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3896)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1792)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3928)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1792)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3897)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1824)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3929)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1824)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3898)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1856)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3930)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1856)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3899)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1888)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3931)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1888)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3900)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1920)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3932)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1920)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3901)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1952)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3933)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1952)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3902)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1984)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3934)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1984)]));
  Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3903)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2016)]));
  Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3935)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2016)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3888)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1537)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3920)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1537)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3889)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1569)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3921)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1569)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3890)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1601)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3922)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1601)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3891)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1633)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3923)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1633)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3892)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1665)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3924)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1665)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3893)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1697)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3925)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1697)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3894)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1729)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3926)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1729)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3895)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1761)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3927)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1761)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3896)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1793)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3928)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1793)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3897)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1825)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3929)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1825)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3898)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1857)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3930)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1857)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3899)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1889)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3931)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1889)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3900)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1921)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3932)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1921)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3901)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1953)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3933)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1953)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3902)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1985)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3934)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 1985)]));
  Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3903)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2017)]));
  Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(((((int)threadIdx.x) >> 4) * 96) + 3935)] * kernel_shared[(((((int)threadIdx.x) & 15) * 2) + 2017)]));
  compute[(((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2))] = max((((Conv2dOutput_local[0] + bias[(((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2))]) * bn_scale[(((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2))]) + bn_offset[(((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2))]), 0.000000e+00f);
  compute[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 256)] = max((((Conv2dOutput_local[2] + bias[(((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2))]) * bn_scale[(((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2))]) + bn_offset[(((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2))]), 0.000000e+00f);
  compute[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1)] = max((((Conv2dOutput_local[1] + bias[((((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1)]) * bn_scale[((((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1)]) + bn_offset[((((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1)]), 0.000000e+00f);
  compute[((((((((int)threadIdx.x) >> 4) * 3584) + ((((int)blockIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 257)] = max((((Conv2dOutput_local[3] + bias[((((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1)]) * bn_scale[((((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1)]) + bn_offset[((((((int)blockIdx.x) & 7) * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1)]), 0.000000e+00f);
}

