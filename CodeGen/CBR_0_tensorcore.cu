
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
extern "C" __global__ void __launch_bounds__(512) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(512) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float Conv2dOutput_local[16];
  __shared__ float PaddedInput_shared[2331];
  __shared__ float kernel_shared[9408];
  for (int vthread_x_s = 0; vthread_x_s < 16; ++vthread_x_s) {
    Conv2dOutput_local[vthread_x_s] = 0.000000e+00f;
  }
  for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 2; ++ax0_ax1_ax2_ax3_fused_0) {
    if (((ax0_ax1_ax2_ax3_fused_0 * 512) + ((int)threadIdx.x)) < 777) {
      for (int ax0_ax1_ax2_ax3_fused_2_s = 0; ax0_ax1_ax2_ax3_fused_2_s < 3; ++ax0_ax1_ax2_ax3_fused_2_s) {
        float condval;
        if (((((3 <= (((((int)blockIdx.x) / 14) * 32) + (((ax0_ax1_ax2_ax3_fused_0 * 512) + ((int)threadIdx.x)) / 21))) && ((((((int)blockIdx.x) / 14) * 32) + (((ax0_ax1_ax2_ax3_fused_0 * 512) + ((int)threadIdx.x)) / 21)) < 227)) && (3 <= (((((int)blockIdx.x) % 14) * 16) + (((ax0_ax1_ax2_ax3_fused_0 * 8) + ((int)threadIdx.x)) % 21)))) && ((((((int)blockIdx.x) % 14) * 16) + (((ax0_ax1_ax2_ax3_fused_0 * 8) + ((int)threadIdx.x)) % 21)) < 227))) {
          condval = data[(((((((((int)blockIdx.x) / 14) * 21504) + ((((ax0_ax1_ax2_ax3_fused_0 * 512) + ((int)threadIdx.x)) / 21) * 672)) + ((((int)blockIdx.x) % 14) * 48)) + ((((ax0_ax1_ax2_ax3_fused_0 * 8) + ((int)threadIdx.x)) % 21) * 3)) + ax0_ax1_ax2_ax3_fused_2_s) - 2025)];
        } else {
          condval = 0.000000e+00f;
        }
        PaddedInput_shared[(((ax0_ax1_ax2_ax3_fused_0 * 1536) + (((int)threadIdx.x) * 3)) + ax0_ax1_ax2_ax3_fused_2_s)] = condval;
      }
    }
  }
  for (int ax0_ax1_ax2_ax3_fused_0_1 = 0; ax0_ax1_ax2_ax3_fused_0_1 < 7; ++ax0_ax1_ax2_ax3_fused_0_1) {
    if (((ax0_ax1_ax2_ax3_fused_0_1 * 8) + (((int)threadIdx.x) >> 6)) < 49) {
        int3 __1;
          int3 __2;
            int3 v_ = make_int3(((ax0_ax1_ax2_ax3_fused_0_1 * 1536) + ((((int)threadIdx.x) >> 6) * 192)), ((ax0_ax1_ax2_ax3_fused_0_1 * 1536) + ((((int)threadIdx.x) >> 6) * 192)), ((ax0_ax1_ax2_ax3_fused_0_1 * 1536) + ((((int)threadIdx.x) >> 6) * 192)));
            int3 __3;
              int3 __4;
                int3 v__1 = make_int3((((((int)threadIdx.x) & 63) * 3))+(1*0), (((((int)threadIdx.x) & 63) * 3))+(1*1), (((((int)threadIdx.x) & 63) * 3))+(1*2));
                int3 v__2 = make_int3(64, 64, 64);
                __4.x = (v__1.x/v__2.x);
                __4.y = (v__1.y/v__2.y);
                __4.z = (v__1.z/v__2.z);
              int3 v__3 = make_int3(64, 64, 64);
              __3.x = (__4.x*v__3.x);
              __3.y = (__4.y*v__3.y);
              __3.z = (__4.z*v__3.z);
            __2.x = (v_.x+__3.x);
            __2.y = (v_.y+__3.y);
            __2.z = (v_.z+__3.z);
          int3 __5;
            int3 v__4 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
            int3 v__5 = make_int3(64, 64, 64);
            __5.x = (v__4.x%v__5.x);
            __5.y = (v__4.y%v__5.y);
            __5.z = (v__4.z%v__5.z);
          __1.x = (__2.x+__5.x);
          __1.y = (__2.y+__5.y);
          __1.z = (__2.z+__5.z);
        int3 __6;
          int3 __7;
            int3 v__6 = make_int3(((ax0_ax1_ax2_ax3_fused_0_1 * 1536) + ((((int)threadIdx.x) >> 6) * 192)), ((ax0_ax1_ax2_ax3_fused_0_1 * 1536) + ((((int)threadIdx.x) >> 6) * 192)), ((ax0_ax1_ax2_ax3_fused_0_1 * 1536) + ((((int)threadIdx.x) >> 6) * 192)));
            int3 __8;
              int3 __9;
                int3 v__7 = make_int3((((((int)threadIdx.x) & 63) * 3))+(1*0), (((((int)threadIdx.x) & 63) * 3))+(1*1), (((((int)threadIdx.x) & 63) * 3))+(1*2));
                int3 v__8 = make_int3(64, 64, 64);
                __9.x = (v__7.x/v__8.x);
                __9.y = (v__7.y/v__8.y);
                __9.z = (v__7.z/v__8.z);
              int3 v__9 = make_int3(64, 64, 64);
              __8.x = (__9.x*v__9.x);
              __8.y = (__9.y*v__9.y);
              __8.z = (__9.z*v__9.z);
            __7.x = (v__6.x+__8.x);
            __7.y = (v__6.y+__8.y);
            __7.z = (v__6.z+__8.z);
          int3 __10;
            int3 v__10 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
            int3 v__11 = make_int3(64, 64, 64);
            __10.x = (v__10.x%v__11.x);
            __10.y = (v__10.y%v__11.y);
            __10.z = (v__10.z%v__11.z);
          __6.x = (__7.x+__10.x);
          __6.y = (__7.y+__10.y);
          __6.z = (__7.z+__10.z);
        float3 v__12 = make_float3(kernel[__6.x],kernel[__6.y],kernel[__6.z]);
        kernel_shared[__1.x] = v__12.x;
        kernel_shared[__1.y] = v__12.y;
        kernel_shared[__1.z] = v__12.z;
    }
  }
  __syncthreads();
  for (int ry_1 = 0; ry_1 < 7; ++ry_1) {
    for (int rx_1 = 0; rx_1 < 7; ++rx_1) {
      for (int rc_2 = 0; rc_2 < 3; ++rc_2) {
        for (int vthread_x_s_1 = 0; vthread_x_s_1 < 16; ++vthread_x_s_1) {
          Conv2dOutput_local[vthread_x_s_1] = (Conv2dOutput_local[vthread_x_s_1] + (PaddedInput_shared[(((((((((int)threadIdx.x) >> 5) * 126) + (ry_1 * 63)) + ((vthread_x_s_1 >> 2) * 12)) + (((((int)threadIdx.x) & 31) >> 4) * 6)) + (rx_1 * 3)) + rc_2)] * kernel_shared[(((((ry_1 * 1344) + (rx_1 * 192)) + (rc_2 * 64)) + ((vthread_x_s_1 & 3) * 16)) + (((int)threadIdx.x) & 15))]));
        }
      }
    }
  }
  for (int vthread_x_s_2 = 0; vthread_x_s_2 < 16; ++vthread_x_s_2) {
    compute[((((((((((int)blockIdx.x) / 14) * 114688) + ((((int)threadIdx.x) >> 5) * 7168)) + ((((int)blockIdx.x) % 14) * 512)) + ((vthread_x_s_2 >> 2) * 128)) + (((((int)threadIdx.x) & 31) >> 4) * 64)) + ((vthread_x_s_2 & 3) * 16)) + (((int)threadIdx.x) & 15))] = max((((Conv2dOutput_local[vthread_x_s_2] + bias[(((vthread_x_s_2 & 3) * 16) + (((int)threadIdx.x) & 15))]) * bn_scale[(((vthread_x_s_2 & 3) * 16) + (((int)threadIdx.x) & 15))]) + bn_offset[(((vthread_x_s_2 & 3) * 16) + (((int)threadIdx.x) & 15))]), 0.000000e+00f);
  }
}

