#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

struct __align__(8) half4 {
  __half x, y, z, w;
  __host__ __device__ half4() : x(__half(0)), y(__half(0)), z(__half(0)), w(__half(0)) {}
  __host__ __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

};
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}

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
extern "C" __global__ void __launch_bounds__(196) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(196) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight) {
  half conv2d_nhwc_local[2];
  __shared__ half PadInput_shared[1296];
  __shared__ half weight_shared[1152];
  conv2d_nhwc_local[0] = __float2half_rn(0.000000e+00f);
  conv2d_nhwc_local[1] = __float2half_rn(0.000000e+00f);
  for (int rc_0 = 0; rc_0 < 8; ++rc_0) {
    __syncthreads();
    half condval;
    if ((((72 <= ((int)threadIdx.x)) && (8 <= (((int)threadIdx.x) % 72))) && ((((int)threadIdx.x) % 72) < 64))) {
      condval = inputs[(((((((((int)threadIdx.x) / 72) * 3584) + (((((int)threadIdx.x) % 72) >> 3) * 512)) + ((((int)blockIdx.x) >> 4) * 128)) + (rc_0 * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 4096)];
    } else {
      condval = __float2half_rn(0.000000e+00f);
    }
    PadInput_shared[(((int)threadIdx.x) * 2)] = condval;
    half condval_1;
    if ((((72 <= ((int)threadIdx.x)) && (8 <= (((int)threadIdx.x) % 72))) && ((((int)threadIdx.x) % 72) < 64))) {
      condval_1 = inputs[(((((((((int)threadIdx.x) / 72) * 3584) + (((((int)threadIdx.x) % 72) >> 3) * 512)) + ((((int)blockIdx.x) >> 4) * 128)) + (rc_0 * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 4095)];
    } else {
      condval_1 = __float2half_rn(0.000000e+00f);
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 1)] = condval_1;
    half condval_2;
    if (((2 <= (((((int)threadIdx.x) >> 2) + 13) % 18)) && ((((((int)threadIdx.x) * 2) + 104) % 144) < 128))) {
      condval_2 = inputs[((((((((((int)threadIdx.x) + 196) / 72) * 3584) + (((((((int)threadIdx.x) >> 2) + 13) % 18) >> 1) * 512)) + ((((int)blockIdx.x) >> 4) * 128)) + (rc_0 * 16)) + (((((int)threadIdx.x) * 2) + 8) & 15)) - 4096)];
    } else {
      condval_2 = __float2half_rn(0.000000e+00f);
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 392)] = condval_2;
    half condval_3;
    if (((2 <= (((((int)threadIdx.x) >> 2) + 13) % 18)) && ((((((int)threadIdx.x) * 2) + 105) % 144) < 128))) {
      condval_3 = inputs[((((((((((int)threadIdx.x) + 196) / 72) * 3584) + (((((((int)threadIdx.x) >> 2) + 13) % 18) >> 1) * 512)) + ((((int)blockIdx.x) >> 4) * 128)) + (rc_0 * 16)) + (((((int)threadIdx.x) * 2) + 9) & 15)) - 4096)];
    } else {
      condval_3 = __float2half_rn(0.000000e+00f);
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 393)] = condval_3;
    half condval_4;
    if ((((((int)threadIdx.x) < 184) && (1 <= (((((int)threadIdx.x) >> 3) + 4) % 9))) && ((((((int)threadIdx.x) * 2) + 64) % 144) < 128))) {
      condval_4 = inputs[((((((((((int)threadIdx.x) + 392) / 72) * 3584) + ((((((int)threadIdx.x) >> 3) + 4) % 9) * 512)) + ((((int)blockIdx.x) >> 4) * 128)) + (rc_0 * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 4096)];
    } else {
      condval_4 = __float2half_rn(0.000000e+00f);
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 784)] = condval_4;
    half condval_5;
    if ((((((int)threadIdx.x) < 184) && (1 <= (((((int)threadIdx.x) >> 3) + 4) % 9))) && ((((((int)threadIdx.x) * 2) + 65) % 144) < 128))) {
      condval_5 = inputs[((((((((((int)threadIdx.x) + 392) / 72) * 3584) + ((((((int)threadIdx.x) >> 3) + 4) % 9) * 512)) + ((((int)blockIdx.x) >> 4) * 128)) + (rc_0 * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 4095)];
    } else {
      condval_5 = __float2half_rn(0.000000e+00f);
    }
    PadInput_shared[((((int)threadIdx.x) * 2) + 785)] = condval_5;
    if (((int)threadIdx.x) < 60) {
      PadInput_shared[((((int)threadIdx.x) * 2) + 1176)] = __float2half_rn(0.000000e+00f);
      PadInput_shared[((((int)threadIdx.x) * 2) + 1177)] = __float2half_rn(0.000000e+00f);
    }
    *(half2*)(weight_shared + (((int)threadIdx.x) * 2)) = *(half2*)(weight + ((((((((int)threadIdx.x) >> 6) * 65536) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 2) * 512)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    *(half2*)(weight_shared + ((((int)threadIdx.x) * 2) + 392)) = *(half2*)(weight + (((((((((int)threadIdx.x) + 196) >> 6) * 65536) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 2) + 1) & 15) * 512)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    if (((int)threadIdx.x) < 184) {
      *(half2*)(weight_shared + ((((int)threadIdx.x) * 2) + 784)) = *(half2*)(weight + (((((((((int)threadIdx.x) + 392) >> 6) * 65536) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 2) + 2) & 15) * 512)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    __syncthreads();
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16))] * weight_shared[((((int)threadIdx.x) & 3) * 2)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16))] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 144)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 384)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 144)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 385)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 288)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 768)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 288)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 769)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 1)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 8)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 1)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 9)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 145)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 392)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 145)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 393)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 289)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 776)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 289)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 777)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 2)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 16)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 2)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 17)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 146)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 400)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 146)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 401)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 290)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 784)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 290)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 785)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 3)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 24)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 3)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 25)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 147)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 408)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 147)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 409)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 291)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 792)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 291)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 793)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 4)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 32)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 4)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 33)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 148)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 416)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 148)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 417)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 292)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 800)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 292)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 801)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 5)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 40)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 5)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 41)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 149)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 424)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 149)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 425)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 293)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 808)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 293)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 809)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 6)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 48)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 6)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 49)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 150)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 432)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 150)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 433)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 294)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 816)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 294)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 817)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 7)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 56)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 7)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 57)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 151)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 440)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 151)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 441)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 295)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 824)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 295)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 825)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 8)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 64)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 8)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 65)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 152)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 448)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 152)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 449)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 296)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 832)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 296)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 833)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 9)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 72)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 9)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 73)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 153)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 456)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 153)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 457)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 297)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 840)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 297)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 841)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 10)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 80)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 10)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 81)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 154)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 464)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 154)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 465)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 298)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 848)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 298)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 849)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 11)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 88)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 11)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 89)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 155)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 472)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 155)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 473)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 299)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 856)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 299)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 857)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 12)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 96)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 12)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 97)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 156)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 480)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 156)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 481)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 300)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 864)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 300)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 865)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 13)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 104)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 13)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 105)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 157)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 488)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 157)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 489)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 301)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 872)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 301)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 873)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 14)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 112)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 14)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 113)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 158)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 496)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 158)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 497)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 302)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 880)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 302)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 881)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 15)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 120)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 15)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 121)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 159)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 504)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 159)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 505)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 303)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 888)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 303)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 889)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 16)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 128)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 16)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 129)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 160)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 512)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 160)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 513)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 304)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 896)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 304)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 897)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 17)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 136)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 17)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 137)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 161)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 520)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 161)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 521)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 305)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 904)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 305)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 905)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 18)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 144)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 18)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 145)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 162)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 528)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 162)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 529)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 306)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 912)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 306)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 913)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 19)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 152)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 19)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 153)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 163)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 536)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 163)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 537)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 307)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 920)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 307)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 921)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 20)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 160)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 20)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 161)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 164)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 544)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 164)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 545)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 308)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 928)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 308)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 929)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 21)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 168)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 21)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 169)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 165)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 552)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 165)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 553)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 309)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 936)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 309)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 937)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 22)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 176)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 22)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 177)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 166)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 560)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 166)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 561)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 310)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 944)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 310)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 945)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 23)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 184)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 23)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 185)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 167)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 568)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 167)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 569)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 311)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 952)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 311)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 953)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 24)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 192)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 24)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 193)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 168)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 576)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 168)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 577)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 312)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 960)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 312)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 961)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 25)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 200)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 25)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 201)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 169)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 584)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 169)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 585)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 313)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 968)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 313)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 969)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 26)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 208)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 26)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 209)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 170)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 592)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 170)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 593)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 314)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 976)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 314)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 977)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 27)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 216)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 27)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 217)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 171)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 600)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 171)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 601)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 315)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 984)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 315)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 985)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 28)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 224)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 28)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 225)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 172)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 608)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 172)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 609)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 316)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 992)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 316)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 993)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 29)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 232)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 29)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 233)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 173)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 616)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 173)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 617)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 317)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1000)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 317)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1001)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 30)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 240)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 30)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 241)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 174)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 624)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 174)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 625)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 318)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1008)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 318)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1009)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 31)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 248)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 31)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 249)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 175)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 632)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 175)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 633)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 319)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1016)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 319)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1017)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 32)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 256)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 32)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 257)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 176)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 640)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 176)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 641)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 320)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1024)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 320)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1025)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 33)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 264)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 33)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 265)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 177)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 648)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 177)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 649)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 321)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1032)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 321)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1033)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 34)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 272)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 34)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 273)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 178)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 656)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 178)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 657)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 322)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1040)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 322)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1041)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 35)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 280)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 35)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 281)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 179)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 664)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 179)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 665)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 323)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1048)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 323)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1049)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 36)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 288)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 36)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 289)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 180)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 672)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 180)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 673)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 324)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1056)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 324)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1057)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 37)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 296)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 37)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 297)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 181)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 680)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 181)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 681)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 325)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1064)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 325)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1065)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 38)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 304)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 38)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 305)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 182)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 688)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 182)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 689)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 326)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1072)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 326)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1073)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 39)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 312)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 39)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 313)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 183)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 696)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 183)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 697)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 327)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1080)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 327)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1081)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 40)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 320)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 40)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 321)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 184)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 704)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 184)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 705)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 328)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1088)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 328)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1089)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 41)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 328)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 41)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 329)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 185)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 712)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 185)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 713)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 329)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1096)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 329)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1097)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 42)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 336)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 42)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 337)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 186)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 720)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 186)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 721)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 330)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1104)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 330)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1105)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 43)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 344)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 43)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 345)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 187)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 728)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 187)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 729)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 331)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1112)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 331)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1113)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 44)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 352)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 44)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 353)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 188)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 736)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 188)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 737)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 332)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1120)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 332)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1121)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 45)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 360)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 45)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 361)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 189)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 744)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 189)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 745)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 333)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1128)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 333)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1129)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 46)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 368)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 46)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 369)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 190)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 752)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 190)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 753)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 334)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1136)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 334)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1137)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 47)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 376)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 47)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 377)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 191)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 760)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 191)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 761)]));
    conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 335)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1144)]));
    conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) / 28) * 144) + (((((int)threadIdx.x) % 28) >> 2) * 16)) + 335)] * weight_shared[(((((int)threadIdx.x) & 3) * 2) + 1145)]));
  }
  conv2d_nhwc[((((((int)threadIdx.x) >> 2) * 512) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((int)threadIdx.x) >> 2) * 512) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)] = conv2d_nhwc_local[1];
}

