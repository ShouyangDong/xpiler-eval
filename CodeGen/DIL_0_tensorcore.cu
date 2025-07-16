
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
extern "C" __global__ void __launch_bounds__(218) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(218) main_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ inputs, float* __restrict__ weight) {
  float conv2d_nhwc_local[8];
  __shared__ float PadInput_shared[687];
  __shared__ float weight_shared[336];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  for (int rh_0 = 0; rh_0 < 7; ++rh_0) {
    __syncthreads();
    float condval;
    if ((((1 < ((((int)blockIdx.x) >> 2) + rh_0)) && (((((int)blockIdx.x) >> 2) + rh_0) < 114)) && (9 <= ((int)threadIdx.x)))) {
      condval = inputs[(((((((int)blockIdx.x) >> 2) * 1344) + (rh_0 * 1344)) + ((int)threadIdx.x)) - 2025)];
    } else {
      condval = 0.000000e+00f;
    }
    PadInput_shared[((int)threadIdx.x)] = condval;
    float condval_1;
    if (((1 < ((((int)blockIdx.x) >> 2) + rh_0)) && (((((int)blockIdx.x) >> 2) + rh_0) < 114))) {
      condval_1 = inputs[((((((((int)blockIdx.x) >> 2) * 1344) + (rh_0 * 1344)) + (((((int)threadIdx.x) + 218) / 3) * 3)) + ((((int)threadIdx.x) + 2) % 3)) - 2025)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PadInput_shared[((((((int)threadIdx.x) + 218) / 3) * 3) + ((((int)threadIdx.x) + 2) % 3))] = condval_1;
    float condval_2;
    if (((1 < ((((int)blockIdx.x) >> 2) + rh_0)) && (((((int)blockIdx.x) >> 2) + rh_0) < 114))) {
      condval_2 = inputs[((((((((int)blockIdx.x) >> 2) * 1344) + (rh_0 * 1344)) + (((((int)threadIdx.x) + 436) / 3) * 3)) + ((((int)threadIdx.x) + 1) % 3)) - 2025)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PadInput_shared[((((((int)threadIdx.x) + 436) / 3) * 3) + ((((int)threadIdx.x) + 1) % 3))] = condval_2;
    if (((int)threadIdx.x) < 33) {
      float condval_3;
      if ((((1 < ((((int)blockIdx.x) >> 2) + rh_0)) && (((((int)blockIdx.x) >> 2) + rh_0) < 114)) && (((int)threadIdx.x) < 27))) {
        condval_3 = inputs[(((((((int)blockIdx.x) >> 2) * 1344) + (rh_0 * 1344)) + ((int)threadIdx.x)) - 1371)];
      } else {
        condval_3 = 0.000000e+00f;
      }
      PadInput_shared[(((int)threadIdx.x) + 654)] = condval_3;
    }
    if (((int)threadIdx.x) < 112) {
        int3 __1;
          int3 __2;
            int3 v_ = make_int3(((((int)threadIdx.x) >> 4) * 48), ((((int)threadIdx.x) >> 4) * 48), ((((int)threadIdx.x) >> 4) * 48));
            int3 __3;
              int3 __4;
                int3 v__1 = make_int3((((((int)threadIdx.x) & 15) * 3))+(1*0), (((((int)threadIdx.x) & 15) * 3))+(1*1), (((((int)threadIdx.x) & 15) * 3))+(1*2));
                int3 v__2 = make_int3(16, 16, 16);
                __4.x = (v__1.x/v__2.x);
                __4.y = (v__1.y/v__2.y);
                __4.z = (v__1.z/v__2.z);
              int3 v__3 = make_int3(16, 16, 16);
              __3.x = (__4.x*v__3.x);
              __3.y = (__4.y*v__3.y);
              __3.z = (__4.z*v__3.z);
            __2.x = (v_.x+__3.x);
            __2.y = (v_.y+__3.y);
            __2.z = (v_.z+__3.z);
          int3 __5;
            int3 v__4 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
            int3 v__5 = make_int3(16, 16, 16);
            __5.x = (v__4.x%v__5.x);
            __5.y = (v__4.y%v__5.y);
            __5.z = (v__4.z%v__5.z);
          __1.x = (__2.x+__5.x);
          __1.y = (__2.y+__5.y);
          __1.z = (__2.z+__5.z);
        int3 __6;
          int3 __7;
            int3 __8;
              int3 v__6 = make_int3(((rh_0 * 1344) + ((((int)threadIdx.x) >> 4) * 192)), ((rh_0 * 1344) + ((((int)threadIdx.x) >> 4) * 192)), ((rh_0 * 1344) + ((((int)threadIdx.x) >> 4) * 192)));
              int3 __9;
                int3 __10;
                  int3 v__7 = make_int3((((((int)threadIdx.x) & 15) * 3))+(1*0), (((((int)threadIdx.x) & 15) * 3))+(1*1), (((((int)threadIdx.x) & 15) * 3))+(1*2));
                  int3 v__8 = make_int3(16, 16, 16);
                  __10.x = (v__7.x/v__8.x);
                  __10.y = (v__7.y/v__8.y);
                  __10.z = (v__7.z/v__8.z);
                int3 v__9 = make_int3(64, 64, 64);
                __9.x = (__10.x*v__9.x);
                __9.y = (__10.y*v__9.y);
                __9.z = (__10.z*v__9.z);
              __8.x = (v__6.x+__9.x);
              __8.y = (v__6.y+__9.y);
              __8.z = (v__6.z+__9.z);
            int3 v__10 = make_int3(((((int)blockIdx.x) & 3) * 16), ((((int)blockIdx.x) & 3) * 16), ((((int)blockIdx.x) & 3) * 16));
            __7.x = (__8.x+v__10.x);
            __7.y = (__8.y+v__10.y);
            __7.z = (__8.z+v__10.z);
          int3 __11;
            int3 v__11 = make_int3(((((int)threadIdx.x) * 3))+(1*0), ((((int)threadIdx.x) * 3))+(1*1), ((((int)threadIdx.x) * 3))+(1*2));
            int3 v__12 = make_int3(16, 16, 16);
            __11.x = (v__11.x%v__12.x);
            __11.y = (v__11.y%v__12.y);
            __11.z = (v__11.z%v__12.z);
          __6.x = (__7.x+__11.x);
          __6.y = (__7.y+__11.y);
          __6.z = (__7.z+__11.z);
        float3 v__13 = make_float3(weight[__6.x],weight[__6.y],weight[__6.z]);
        weight_shared[__1.x] = v__13.x;
        weight_shared[__1.y] = v__13.y;
        weight_shared[__1.z] = v__13.z;
    }
    __syncthreads();
    for (int co_3 = 0; co_3 < 4; ++co_3) {
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[((((int)threadIdx.x) >> 1) * 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 8) + (co_3 * 2))]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[((((int)threadIdx.x) >> 1) * 6)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 1)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 1)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 16)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 1)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 17)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 2)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 32)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 2)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 33)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 6)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 48)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 6)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 49)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 7)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 64)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 7)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 65)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 8)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 80)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 8)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 81)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 12)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 96)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 12)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 97)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 13)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 112)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 13)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 113)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 14)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 128)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 14)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 129)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 18)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 144)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 18)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 145)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 19)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 160)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 19)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 161)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 20)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 176)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 20)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 177)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 24)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 192)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 24)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 193)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 25)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 208)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 25)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 209)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 26)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 224)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 26)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 225)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 30)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 240)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 30)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 241)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 31)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 256)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 31)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 257)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 32)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 272)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 32)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 273)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 36)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 288)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 36)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 289)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 37)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 304)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 37)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 305)]));
      conv2d_nhwc_local[(co_3 * 2)] = (conv2d_nhwc_local[(co_3 * 2)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 38)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 320)]));
      conv2d_nhwc_local[((co_3 * 2) + 1)] = (conv2d_nhwc_local[((co_3 * 2) + 1)] + (PadInput_shared[(((((int)threadIdx.x) >> 1) * 6) + 38)] * weight_shared[((((((int)threadIdx.x) & 1) * 8) + (co_3 * 2)) + 321)]));
    }
  }
  conv2d_nhwc[(((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8))] = conv2d_nhwc_local[0];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1)] = conv2d_nhwc_local[1];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 2)] = conv2d_nhwc_local[2];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 3)] = conv2d_nhwc_local[3];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 4)] = conv2d_nhwc_local[4];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 5)] = conv2d_nhwc_local[5];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 6)] = conv2d_nhwc_local[6];
  conv2d_nhwc[((((((((int)blockIdx.x) >> 2) * 6976) + ((((int)threadIdx.x) >> 1) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 7)] = conv2d_nhwc_local[7];
}

