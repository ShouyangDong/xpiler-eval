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
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ conv2d_transpose_nhwc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ conv2d_transpose_nhwc, half* __restrict__ inputs, half* __restrict__ weight) {
  half conv2d_transpose_nhwc_local[2];
  __shared__ half PadInput_shared[512];
  __shared__ half weight_shared[8192];
  conv2d_transpose_nhwc_local[0] = __float2half_rn(0.000000e+00f);
  conv2d_transpose_nhwc_local[1] = __float2half_rn(0.000000e+00f);
  for (int rc_0 = 0; rc_0 < 16; ++rc_0) {
    __syncthreads();
    half4 condval;
    if (((((1 <= (((((int)blockIdx.x) >> 5) * 2) + (((int)threadIdx.x) >> 5))) && ((((((int)blockIdx.x) >> 5) * 2) + (((int)threadIdx.x) >> 5)) < 5)) && (1 <= ((((((int)blockIdx.x) & 31) >> 4) * 2) + ((((int)threadIdx.x) & 31) >> 3)))) && (((((((int)blockIdx.x) & 31) >> 4) * 2) + ((((int)threadIdx.x) & 31) >> 3)) < 5))) {
      condval = *(half4*)(inputs + (((((((((int)blockIdx.x) >> 5) * 4096) + (((((int)blockIdx.x) & 31) >> 4) * 1024)) + ((((int)threadIdx.x) >> 3) * 512)) + (rc_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) - 2560));
    } else {
      condval = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
    }
    *(half4*)(PadInput_shared + (((int)threadIdx.x) * 4)) = condval;
    *(uint4*)(weight_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(weight + ((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4*)(weight + (((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 262144));
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 2048)) = *(uint4*)(weight + (((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 524288));
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 3072)) = *(uint4*)(weight + (((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 786432));
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 4096)) = *(uint4*)(weight + (((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1048576));
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 5120)) = *(uint4*)(weight + (((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1310720));
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 6144)) = *(uint4*)(weight + (((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1572864));
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 7168)) = *(uint4*)(weight + (((((((((int)threadIdx.x) >> 6) * 131072) + (rc_0 * 8192)) + (((((int)threadIdx.x) & 63) >> 1) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1835008));
    __syncthreads();
    for (int rw_1 = 0; rw_1 < 2; ++rw_1) {
      half condval_1;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_1 = PadInput_shared[((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32))];
      } else {
        condval_1 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_1 * weight_shared[(((((int)threadIdx.x) & 15) + 7680) - (rw_1 * 1024))]));
      half condval_2;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_2 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 1)];
      } else {
        condval_2 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_2 * weight_shared[(((((int)threadIdx.x) & 15) + 7696) - (rw_1 * 1024))]));
      half condval_3;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_3 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 2)];
      } else {
        condval_3 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_3 * weight_shared[(((((int)threadIdx.x) & 15) + 7712) - (rw_1 * 1024))]));
      half condval_4;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_4 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 3)];
      } else {
        condval_4 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_4 * weight_shared[(((((int)threadIdx.x) & 15) + 7728) - (rw_1 * 1024))]));
      half condval_5;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_5 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 4)];
      } else {
        condval_5 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_5 * weight_shared[(((((int)threadIdx.x) & 15) + 7744) - (rw_1 * 1024))]));
      half condval_6;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_6 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 5)];
      } else {
        condval_6 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_6 * weight_shared[(((((int)threadIdx.x) & 15) + 7760) - (rw_1 * 1024))]));
      half condval_7;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_7 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 6)];
      } else {
        condval_7 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_7 * weight_shared[(((((int)threadIdx.x) & 15) + 7776) - (rw_1 * 1024))]));
      half condval_8;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_8 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 7)];
      } else {
        condval_8 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_8 * weight_shared[(((((int)threadIdx.x) & 15) + 7792) - (rw_1 * 1024))]));
      half condval_9;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_9 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 8)];
      } else {
        condval_9 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_9 * weight_shared[(((((int)threadIdx.x) & 15) + 7808) - (rw_1 * 1024))]));
      half condval_10;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_10 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 9)];
      } else {
        condval_10 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_10 * weight_shared[(((((int)threadIdx.x) & 15) + 7824) - (rw_1 * 1024))]));
      half condval_11;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_11 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 10)];
      } else {
        condval_11 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_11 * weight_shared[(((((int)threadIdx.x) & 15) + 7840) - (rw_1 * 1024))]));
      half condval_12;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_12 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 11)];
      } else {
        condval_12 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_12 * weight_shared[(((((int)threadIdx.x) & 15) + 7856) - (rw_1 * 1024))]));
      half condval_13;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_13 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 12)];
      } else {
        condval_13 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_13 * weight_shared[(((((int)threadIdx.x) & 15) + 7872) - (rw_1 * 1024))]));
      half condval_14;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_14 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 13)];
      } else {
        condval_14 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_14 * weight_shared[(((((int)threadIdx.x) & 15) + 7888) - (rw_1 * 1024))]));
      half condval_15;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_15 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 14)];
      } else {
        condval_15 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_15 * weight_shared[(((((int)threadIdx.x) & 15) + 7904) - (rw_1 * 1024))]));
      half condval_16;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_16 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 15)];
      } else {
        condval_16 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_16 * weight_shared[(((((int)threadIdx.x) & 15) + 7920) - (rw_1 * 1024))]));
      half condval_17;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_17 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 16)];
      } else {
        condval_17 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_17 * weight_shared[(((((int)threadIdx.x) & 15) + 7936) - (rw_1 * 1024))]));
      half condval_18;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_18 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 17)];
      } else {
        condval_18 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_18 * weight_shared[(((((int)threadIdx.x) & 15) + 7952) - (rw_1 * 1024))]));
      half condval_19;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_19 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 18)];
      } else {
        condval_19 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_19 * weight_shared[(((((int)threadIdx.x) & 15) + 7968) - (rw_1 * 1024))]));
      half condval_20;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_20 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 19)];
      } else {
        condval_20 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_20 * weight_shared[(((((int)threadIdx.x) & 15) + 7984) - (rw_1 * 1024))]));
      half condval_21;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_21 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 20)];
      } else {
        condval_21 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_21 * weight_shared[(((((int)threadIdx.x) & 15) + 8000) - (rw_1 * 1024))]));
      half condval_22;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_22 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 21)];
      } else {
        condval_22 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_22 * weight_shared[(((((int)threadIdx.x) & 15) + 8016) - (rw_1 * 1024))]));
      half condval_23;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_23 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 22)];
      } else {
        condval_23 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_23 * weight_shared[(((((int)threadIdx.x) & 15) + 8032) - (rw_1 * 1024))]));
      half condval_24;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_24 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 23)];
      } else {
        condval_24 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_24 * weight_shared[(((((int)threadIdx.x) & 15) + 8048) - (rw_1 * 1024))]));
      half condval_25;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_25 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 24)];
      } else {
        condval_25 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_25 * weight_shared[(((((int)threadIdx.x) & 15) + 8064) - (rw_1 * 1024))]));
      half condval_26;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_26 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 25)];
      } else {
        condval_26 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_26 * weight_shared[(((((int)threadIdx.x) & 15) + 8080) - (rw_1 * 1024))]));
      half condval_27;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_27 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 26)];
      } else {
        condval_27 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_27 * weight_shared[(((((int)threadIdx.x) & 15) + 8096) - (rw_1 * 1024))]));
      half condval_28;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_28 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 27)];
      } else {
        condval_28 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_28 * weight_shared[(((((int)threadIdx.x) & 15) + 8112) - (rw_1 * 1024))]));
      half condval_29;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_29 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 28)];
      } else {
        condval_29 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_29 * weight_shared[(((((int)threadIdx.x) & 15) + 8128) - (rw_1 * 1024))]));
      half condval_30;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_30 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 29)];
      } else {
        condval_30 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_30 * weight_shared[(((((int)threadIdx.x) & 15) + 8144) - (rw_1 * 1024))]));
      half condval_31;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_31 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 30)];
      } else {
        condval_31 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_31 * weight_shared[(((((int)threadIdx.x) & 15) + 8160) - (rw_1 * 1024))]));
      half condval_32;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_32 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 31)];
      } else {
        condval_32 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_32 * weight_shared[(((((int)threadIdx.x) & 15) + 8176) - (rw_1 * 1024))]));
      half condval_33;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_33 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 32)];
      } else {
        condval_33 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_33 * weight_shared[(((((int)threadIdx.x) & 15) + 7168) - (rw_1 * 1024))]));
      half condval_34;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_34 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 33)];
      } else {
        condval_34 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_34 * weight_shared[(((((int)threadIdx.x) & 15) + 7184) - (rw_1 * 1024))]));
      half condval_35;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_35 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 34)];
      } else {
        condval_35 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_35 * weight_shared[(((((int)threadIdx.x) & 15) + 7200) - (rw_1 * 1024))]));
      half condval_36;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_36 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 35)];
      } else {
        condval_36 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_36 * weight_shared[(((((int)threadIdx.x) & 15) + 7216) - (rw_1 * 1024))]));
      half condval_37;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_37 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 36)];
      } else {
        condval_37 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_37 * weight_shared[(((((int)threadIdx.x) & 15) + 7232) - (rw_1 * 1024))]));
      half condval_38;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_38 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 37)];
      } else {
        condval_38 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_38 * weight_shared[(((((int)threadIdx.x) & 15) + 7248) - (rw_1 * 1024))]));
      half condval_39;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_39 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 38)];
      } else {
        condval_39 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_39 * weight_shared[(((((int)threadIdx.x) & 15) + 7264) - (rw_1 * 1024))]));
      half condval_40;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_40 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 39)];
      } else {
        condval_40 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_40 * weight_shared[(((((int)threadIdx.x) & 15) + 7280) - (rw_1 * 1024))]));
      half condval_41;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_41 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 40)];
      } else {
        condval_41 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_41 * weight_shared[(((((int)threadIdx.x) & 15) + 7296) - (rw_1 * 1024))]));
      half condval_42;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_42 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 41)];
      } else {
        condval_42 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_42 * weight_shared[(((((int)threadIdx.x) & 15) + 7312) - (rw_1 * 1024))]));
      half condval_43;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_43 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 42)];
      } else {
        condval_43 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_43 * weight_shared[(((((int)threadIdx.x) & 15) + 7328) - (rw_1 * 1024))]));
      half condval_44;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_44 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 43)];
      } else {
        condval_44 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_44 * weight_shared[(((((int)threadIdx.x) & 15) + 7344) - (rw_1 * 1024))]));
      half condval_45;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_45 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 44)];
      } else {
        condval_45 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_45 * weight_shared[(((((int)threadIdx.x) & 15) + 7360) - (rw_1 * 1024))]));
      half condval_46;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_46 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 45)];
      } else {
        condval_46 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_46 * weight_shared[(((((int)threadIdx.x) & 15) + 7376) - (rw_1 * 1024))]));
      half condval_47;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_47 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 46)];
      } else {
        condval_47 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_47 * weight_shared[(((((int)threadIdx.x) & 15) + 7392) - (rw_1 * 1024))]));
      half condval_48;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_48 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 47)];
      } else {
        condval_48 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_48 * weight_shared[(((((int)threadIdx.x) & 15) + 7408) - (rw_1 * 1024))]));
      half condval_49;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_49 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 48)];
      } else {
        condval_49 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_49 * weight_shared[(((((int)threadIdx.x) & 15) + 7424) - (rw_1 * 1024))]));
      half condval_50;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_50 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 49)];
      } else {
        condval_50 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_50 * weight_shared[(((((int)threadIdx.x) & 15) + 7440) - (rw_1 * 1024))]));
      half condval_51;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_51 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 50)];
      } else {
        condval_51 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_51 * weight_shared[(((((int)threadIdx.x) & 15) + 7456) - (rw_1 * 1024))]));
      half condval_52;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_52 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 51)];
      } else {
        condval_52 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_52 * weight_shared[(((((int)threadIdx.x) & 15) + 7472) - (rw_1 * 1024))]));
      half condval_53;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_53 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 52)];
      } else {
        condval_53 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_53 * weight_shared[(((((int)threadIdx.x) & 15) + 7488) - (rw_1 * 1024))]));
      half condval_54;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_54 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 53)];
      } else {
        condval_54 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_54 * weight_shared[(((((int)threadIdx.x) & 15) + 7504) - (rw_1 * 1024))]));
      half condval_55;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_55 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 54)];
      } else {
        condval_55 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_55 * weight_shared[(((((int)threadIdx.x) & 15) + 7520) - (rw_1 * 1024))]));
      half condval_56;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_56 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 55)];
      } else {
        condval_56 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_56 * weight_shared[(((((int)threadIdx.x) & 15) + 7536) - (rw_1 * 1024))]));
      half condval_57;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_57 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 56)];
      } else {
        condval_57 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_57 * weight_shared[(((((int)threadIdx.x) & 15) + 7552) - (rw_1 * 1024))]));
      half condval_58;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_58 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 57)];
      } else {
        condval_58 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_58 * weight_shared[(((((int)threadIdx.x) & 15) + 7568) - (rw_1 * 1024))]));
      half condval_59;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_59 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 58)];
      } else {
        condval_59 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_59 * weight_shared[(((((int)threadIdx.x) & 15) + 7584) - (rw_1 * 1024))]));
      half condval_60;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_60 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 59)];
      } else {
        condval_60 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_60 * weight_shared[(((((int)threadIdx.x) & 15) + 7600) - (rw_1 * 1024))]));
      half condval_61;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_61 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 60)];
      } else {
        condval_61 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_61 * weight_shared[(((((int)threadIdx.x) & 15) + 7616) - (rw_1 * 1024))]));
      half condval_62;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_62 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 61)];
      } else {
        condval_62 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_62 * weight_shared[(((((int)threadIdx.x) & 15) + 7632) - (rw_1 * 1024))]));
      half condval_63;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_63 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 62)];
      } else {
        condval_63 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_63 * weight_shared[(((((int)threadIdx.x) & 15) + 7648) - (rw_1 * 1024))]));
      half condval_64;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_64 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 63)];
      } else {
        condval_64 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_64 * weight_shared[(((((int)threadIdx.x) & 15) + 7664) - (rw_1 * 1024))]));
      half condval_65;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_65 = PadInput_shared[(((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32))];
      } else {
        condval_65 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_65 * weight_shared[(((((int)threadIdx.x) & 15) + 5632) - (rw_1 * 1024))]));
      half condval_66;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_66 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 1)];
      } else {
        condval_66 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_66 * weight_shared[(((((int)threadIdx.x) & 15) + 5648) - (rw_1 * 1024))]));
      half condval_67;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_67 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 2)];
      } else {
        condval_67 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_67 * weight_shared[(((((int)threadIdx.x) & 15) + 5664) - (rw_1 * 1024))]));
      half condval_68;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_68 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 3)];
      } else {
        condval_68 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_68 * weight_shared[(((((int)threadIdx.x) & 15) + 5680) - (rw_1 * 1024))]));
      half condval_69;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_69 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 4)];
      } else {
        condval_69 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_69 * weight_shared[(((((int)threadIdx.x) & 15) + 5696) - (rw_1 * 1024))]));
      half condval_70;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_70 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 5)];
      } else {
        condval_70 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_70 * weight_shared[(((((int)threadIdx.x) & 15) + 5712) - (rw_1 * 1024))]));
      half condval_71;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_71 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 6)];
      } else {
        condval_71 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_71 * weight_shared[(((((int)threadIdx.x) & 15) + 5728) - (rw_1 * 1024))]));
      half condval_72;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_72 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 7)];
      } else {
        condval_72 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_72 * weight_shared[(((((int)threadIdx.x) & 15) + 5744) - (rw_1 * 1024))]));
      half condval_73;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_73 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 8)];
      } else {
        condval_73 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_73 * weight_shared[(((((int)threadIdx.x) & 15) + 5760) - (rw_1 * 1024))]));
      half condval_74;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_74 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 9)];
      } else {
        condval_74 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_74 * weight_shared[(((((int)threadIdx.x) & 15) + 5776) - (rw_1 * 1024))]));
      half condval_75;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_75 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 10)];
      } else {
        condval_75 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_75 * weight_shared[(((((int)threadIdx.x) & 15) + 5792) - (rw_1 * 1024))]));
      half condval_76;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_76 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 11)];
      } else {
        condval_76 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_76 * weight_shared[(((((int)threadIdx.x) & 15) + 5808) - (rw_1 * 1024))]));
      half condval_77;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_77 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 12)];
      } else {
        condval_77 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_77 * weight_shared[(((((int)threadIdx.x) & 15) + 5824) - (rw_1 * 1024))]));
      half condval_78;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_78 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 13)];
      } else {
        condval_78 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_78 * weight_shared[(((((int)threadIdx.x) & 15) + 5840) - (rw_1 * 1024))]));
      half condval_79;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_79 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 14)];
      } else {
        condval_79 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_79 * weight_shared[(((((int)threadIdx.x) & 15) + 5856) - (rw_1 * 1024))]));
      half condval_80;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_80 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 15)];
      } else {
        condval_80 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_80 * weight_shared[(((((int)threadIdx.x) & 15) + 5872) - (rw_1 * 1024))]));
      half condval_81;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_81 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 16)];
      } else {
        condval_81 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_81 * weight_shared[(((((int)threadIdx.x) & 15) + 5888) - (rw_1 * 1024))]));
      half condval_82;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_82 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 17)];
      } else {
        condval_82 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_82 * weight_shared[(((((int)threadIdx.x) & 15) + 5904) - (rw_1 * 1024))]));
      half condval_83;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_83 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 18)];
      } else {
        condval_83 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_83 * weight_shared[(((((int)threadIdx.x) & 15) + 5920) - (rw_1 * 1024))]));
      half condval_84;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_84 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 19)];
      } else {
        condval_84 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_84 * weight_shared[(((((int)threadIdx.x) & 15) + 5936) - (rw_1 * 1024))]));
      half condval_85;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_85 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 20)];
      } else {
        condval_85 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_85 * weight_shared[(((((int)threadIdx.x) & 15) + 5952) - (rw_1 * 1024))]));
      half condval_86;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_86 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 21)];
      } else {
        condval_86 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_86 * weight_shared[(((((int)threadIdx.x) & 15) + 5968) - (rw_1 * 1024))]));
      half condval_87;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_87 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 22)];
      } else {
        condval_87 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_87 * weight_shared[(((((int)threadIdx.x) & 15) + 5984) - (rw_1 * 1024))]));
      half condval_88;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_88 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 23)];
      } else {
        condval_88 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_88 * weight_shared[(((((int)threadIdx.x) & 15) + 6000) - (rw_1 * 1024))]));
      half condval_89;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_89 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 24)];
      } else {
        condval_89 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_89 * weight_shared[(((((int)threadIdx.x) & 15) + 6016) - (rw_1 * 1024))]));
      half condval_90;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_90 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 25)];
      } else {
        condval_90 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_90 * weight_shared[(((((int)threadIdx.x) & 15) + 6032) - (rw_1 * 1024))]));
      half condval_91;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_91 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 26)];
      } else {
        condval_91 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_91 * weight_shared[(((((int)threadIdx.x) & 15) + 6048) - (rw_1 * 1024))]));
      half condval_92;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_92 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 27)];
      } else {
        condval_92 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_92 * weight_shared[(((((int)threadIdx.x) & 15) + 6064) - (rw_1 * 1024))]));
      half condval_93;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_93 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 28)];
      } else {
        condval_93 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_93 * weight_shared[(((((int)threadIdx.x) & 15) + 6080) - (rw_1 * 1024))]));
      half condval_94;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_94 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 29)];
      } else {
        condval_94 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_94 * weight_shared[(((((int)threadIdx.x) & 15) + 6096) - (rw_1 * 1024))]));
      half condval_95;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_95 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 30)];
      } else {
        condval_95 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_95 * weight_shared[(((((int)threadIdx.x) & 15) + 6112) - (rw_1 * 1024))]));
      half condval_96;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_96 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 31)];
      } else {
        condval_96 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_96 * weight_shared[(((((int)threadIdx.x) & 15) + 6128) - (rw_1 * 1024))]));
      half condval_97;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_97 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 32)];
      } else {
        condval_97 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_97 * weight_shared[(((((int)threadIdx.x) & 15) + 5120) - (rw_1 * 1024))]));
      half condval_98;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_98 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 33)];
      } else {
        condval_98 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_98 * weight_shared[(((((int)threadIdx.x) & 15) + 5136) - (rw_1 * 1024))]));
      half condval_99;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_99 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 34)];
      } else {
        condval_99 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_99 * weight_shared[(((((int)threadIdx.x) & 15) + 5152) - (rw_1 * 1024))]));
      half condval_100;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_100 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 35)];
      } else {
        condval_100 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_100 * weight_shared[(((((int)threadIdx.x) & 15) + 5168) - (rw_1 * 1024))]));
      half condval_101;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_101 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 36)];
      } else {
        condval_101 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_101 * weight_shared[(((((int)threadIdx.x) & 15) + 5184) - (rw_1 * 1024))]));
      half condval_102;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_102 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 37)];
      } else {
        condval_102 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_102 * weight_shared[(((((int)threadIdx.x) & 15) + 5200) - (rw_1 * 1024))]));
      half condval_103;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_103 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 38)];
      } else {
        condval_103 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_103 * weight_shared[(((((int)threadIdx.x) & 15) + 5216) - (rw_1 * 1024))]));
      half condval_104;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_104 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 39)];
      } else {
        condval_104 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_104 * weight_shared[(((((int)threadIdx.x) & 15) + 5232) - (rw_1 * 1024))]));
      half condval_105;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_105 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 40)];
      } else {
        condval_105 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_105 * weight_shared[(((((int)threadIdx.x) & 15) + 5248) - (rw_1 * 1024))]));
      half condval_106;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_106 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 41)];
      } else {
        condval_106 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_106 * weight_shared[(((((int)threadIdx.x) & 15) + 5264) - (rw_1 * 1024))]));
      half condval_107;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_107 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 42)];
      } else {
        condval_107 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_107 * weight_shared[(((((int)threadIdx.x) & 15) + 5280) - (rw_1 * 1024))]));
      half condval_108;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_108 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 43)];
      } else {
        condval_108 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_108 * weight_shared[(((((int)threadIdx.x) & 15) + 5296) - (rw_1 * 1024))]));
      half condval_109;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_109 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 44)];
      } else {
        condval_109 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_109 * weight_shared[(((((int)threadIdx.x) & 15) + 5312) - (rw_1 * 1024))]));
      half condval_110;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_110 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 45)];
      } else {
        condval_110 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_110 * weight_shared[(((((int)threadIdx.x) & 15) + 5328) - (rw_1 * 1024))]));
      half condval_111;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_111 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 46)];
      } else {
        condval_111 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_111 * weight_shared[(((((int)threadIdx.x) & 15) + 5344) - (rw_1 * 1024))]));
      half condval_112;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_112 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 47)];
      } else {
        condval_112 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_112 * weight_shared[(((((int)threadIdx.x) & 15) + 5360) - (rw_1 * 1024))]));
      half condval_113;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_113 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 48)];
      } else {
        condval_113 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_113 * weight_shared[(((((int)threadIdx.x) & 15) + 5376) - (rw_1 * 1024))]));
      half condval_114;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_114 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 49)];
      } else {
        condval_114 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_114 * weight_shared[(((((int)threadIdx.x) & 15) + 5392) - (rw_1 * 1024))]));
      half condval_115;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_115 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 50)];
      } else {
        condval_115 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_115 * weight_shared[(((((int)threadIdx.x) & 15) + 5408) - (rw_1 * 1024))]));
      half condval_116;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_116 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 51)];
      } else {
        condval_116 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_116 * weight_shared[(((((int)threadIdx.x) & 15) + 5424) - (rw_1 * 1024))]));
      half condval_117;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_117 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 52)];
      } else {
        condval_117 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_117 * weight_shared[(((((int)threadIdx.x) & 15) + 5440) - (rw_1 * 1024))]));
      half condval_118;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_118 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 53)];
      } else {
        condval_118 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_118 * weight_shared[(((((int)threadIdx.x) & 15) + 5456) - (rw_1 * 1024))]));
      half condval_119;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_119 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 54)];
      } else {
        condval_119 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_119 * weight_shared[(((((int)threadIdx.x) & 15) + 5472) - (rw_1 * 1024))]));
      half condval_120;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_120 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 55)];
      } else {
        condval_120 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_120 * weight_shared[(((((int)threadIdx.x) & 15) + 5488) - (rw_1 * 1024))]));
      half condval_121;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_121 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 56)];
      } else {
        condval_121 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_121 * weight_shared[(((((int)threadIdx.x) & 15) + 5504) - (rw_1 * 1024))]));
      half condval_122;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_122 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 57)];
      } else {
        condval_122 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_122 * weight_shared[(((((int)threadIdx.x) & 15) + 5520) - (rw_1 * 1024))]));
      half condval_123;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_123 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 58)];
      } else {
        condval_123 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_123 * weight_shared[(((((int)threadIdx.x) & 15) + 5536) - (rw_1 * 1024))]));
      half condval_124;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_124 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 59)];
      } else {
        condval_124 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_124 * weight_shared[(((((int)threadIdx.x) & 15) + 5552) - (rw_1 * 1024))]));
      half condval_125;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_125 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 60)];
      } else {
        condval_125 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_125 * weight_shared[(((((int)threadIdx.x) & 15) + 5568) - (rw_1 * 1024))]));
      half condval_126;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_126 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 61)];
      } else {
        condval_126 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_126 * weight_shared[(((((int)threadIdx.x) & 15) + 5584) - (rw_1 * 1024))]));
      half condval_127;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_127 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 62)];
      } else {
        condval_127 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_127 * weight_shared[(((((int)threadIdx.x) & 15) + 5600) - (rw_1 * 1024))]));
      half condval_128;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_128 = PadInput_shared[((((((((int)threadIdx.x) + 32) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 63)];
      } else {
        condval_128 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_128 * weight_shared[(((((int)threadIdx.x) & 15) + 5616) - (rw_1 * 1024))]));
      half condval_129;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_129 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 128)];
      } else {
        condval_129 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_129 * weight_shared[(((((int)threadIdx.x) & 15) + 3584) - (rw_1 * 1024))]));
      half condval_130;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_130 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 129)];
      } else {
        condval_130 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_130 * weight_shared[(((((int)threadIdx.x) & 15) + 3600) - (rw_1 * 1024))]));
      half condval_131;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_131 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 130)];
      } else {
        condval_131 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_131 * weight_shared[(((((int)threadIdx.x) & 15) + 3616) - (rw_1 * 1024))]));
      half condval_132;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_132 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 131)];
      } else {
        condval_132 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_132 * weight_shared[(((((int)threadIdx.x) & 15) + 3632) - (rw_1 * 1024))]));
      half condval_133;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_133 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 132)];
      } else {
        condval_133 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_133 * weight_shared[(((((int)threadIdx.x) & 15) + 3648) - (rw_1 * 1024))]));
      half condval_134;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_134 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 133)];
      } else {
        condval_134 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_134 * weight_shared[(((((int)threadIdx.x) & 15) + 3664) - (rw_1 * 1024))]));
      half condval_135;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_135 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 134)];
      } else {
        condval_135 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_135 * weight_shared[(((((int)threadIdx.x) & 15) + 3680) - (rw_1 * 1024))]));
      half condval_136;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_136 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 135)];
      } else {
        condval_136 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_136 * weight_shared[(((((int)threadIdx.x) & 15) + 3696) - (rw_1 * 1024))]));
      half condval_137;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_137 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 136)];
      } else {
        condval_137 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_137 * weight_shared[(((((int)threadIdx.x) & 15) + 3712) - (rw_1 * 1024))]));
      half condval_138;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_138 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 137)];
      } else {
        condval_138 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_138 * weight_shared[(((((int)threadIdx.x) & 15) + 3728) - (rw_1 * 1024))]));
      half condval_139;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_139 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 138)];
      } else {
        condval_139 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_139 * weight_shared[(((((int)threadIdx.x) & 15) + 3744) - (rw_1 * 1024))]));
      half condval_140;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_140 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 139)];
      } else {
        condval_140 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_140 * weight_shared[(((((int)threadIdx.x) & 15) + 3760) - (rw_1 * 1024))]));
      half condval_141;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_141 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 140)];
      } else {
        condval_141 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_141 * weight_shared[(((((int)threadIdx.x) & 15) + 3776) - (rw_1 * 1024))]));
      half condval_142;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_142 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 141)];
      } else {
        condval_142 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_142 * weight_shared[(((((int)threadIdx.x) & 15) + 3792) - (rw_1 * 1024))]));
      half condval_143;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_143 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 142)];
      } else {
        condval_143 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_143 * weight_shared[(((((int)threadIdx.x) & 15) + 3808) - (rw_1 * 1024))]));
      half condval_144;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_144 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 143)];
      } else {
        condval_144 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_144 * weight_shared[(((((int)threadIdx.x) & 15) + 3824) - (rw_1 * 1024))]));
      half condval_145;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_145 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 144)];
      } else {
        condval_145 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_145 * weight_shared[(((((int)threadIdx.x) & 15) + 3840) - (rw_1 * 1024))]));
      half condval_146;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_146 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 145)];
      } else {
        condval_146 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_146 * weight_shared[(((((int)threadIdx.x) & 15) + 3856) - (rw_1 * 1024))]));
      half condval_147;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_147 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 146)];
      } else {
        condval_147 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_147 * weight_shared[(((((int)threadIdx.x) & 15) + 3872) - (rw_1 * 1024))]));
      half condval_148;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_148 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 147)];
      } else {
        condval_148 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_148 * weight_shared[(((((int)threadIdx.x) & 15) + 3888) - (rw_1 * 1024))]));
      half condval_149;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_149 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 148)];
      } else {
        condval_149 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_149 * weight_shared[(((((int)threadIdx.x) & 15) + 3904) - (rw_1 * 1024))]));
      half condval_150;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_150 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 149)];
      } else {
        condval_150 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_150 * weight_shared[(((((int)threadIdx.x) & 15) + 3920) - (rw_1 * 1024))]));
      half condval_151;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_151 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 150)];
      } else {
        condval_151 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_151 * weight_shared[(((((int)threadIdx.x) & 15) + 3936) - (rw_1 * 1024))]));
      half condval_152;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_152 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 151)];
      } else {
        condval_152 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_152 * weight_shared[(((((int)threadIdx.x) & 15) + 3952) - (rw_1 * 1024))]));
      half condval_153;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_153 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 152)];
      } else {
        condval_153 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_153 * weight_shared[(((((int)threadIdx.x) & 15) + 3968) - (rw_1 * 1024))]));
      half condval_154;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_154 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 153)];
      } else {
        condval_154 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_154 * weight_shared[(((((int)threadIdx.x) & 15) + 3984) - (rw_1 * 1024))]));
      half condval_155;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_155 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 154)];
      } else {
        condval_155 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_155 * weight_shared[(((((int)threadIdx.x) & 15) + 4000) - (rw_1 * 1024))]));
      half condval_156;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_156 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 155)];
      } else {
        condval_156 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_156 * weight_shared[(((((int)threadIdx.x) & 15) + 4016) - (rw_1 * 1024))]));
      half condval_157;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_157 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 156)];
      } else {
        condval_157 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_157 * weight_shared[(((((int)threadIdx.x) & 15) + 4032) - (rw_1 * 1024))]));
      half condval_158;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_158 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 157)];
      } else {
        condval_158 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_158 * weight_shared[(((((int)threadIdx.x) & 15) + 4048) - (rw_1 * 1024))]));
      half condval_159;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_159 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 158)];
      } else {
        condval_159 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_159 * weight_shared[(((((int)threadIdx.x) & 15) + 4064) - (rw_1 * 1024))]));
      half condval_160;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_160 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 159)];
      } else {
        condval_160 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_160 * weight_shared[(((((int)threadIdx.x) & 15) + 4080) - (rw_1 * 1024))]));
      half condval_161;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_161 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 160)];
      } else {
        condval_161 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_161 * weight_shared[(((((int)threadIdx.x) & 15) + 3072) - (rw_1 * 1024))]));
      half condval_162;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_162 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 161)];
      } else {
        condval_162 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_162 * weight_shared[(((((int)threadIdx.x) & 15) + 3088) - (rw_1 * 1024))]));
      half condval_163;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_163 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 162)];
      } else {
        condval_163 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_163 * weight_shared[(((((int)threadIdx.x) & 15) + 3104) - (rw_1 * 1024))]));
      half condval_164;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_164 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 163)];
      } else {
        condval_164 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_164 * weight_shared[(((((int)threadIdx.x) & 15) + 3120) - (rw_1 * 1024))]));
      half condval_165;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_165 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 164)];
      } else {
        condval_165 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_165 * weight_shared[(((((int)threadIdx.x) & 15) + 3136) - (rw_1 * 1024))]));
      half condval_166;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_166 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 165)];
      } else {
        condval_166 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_166 * weight_shared[(((((int)threadIdx.x) & 15) + 3152) - (rw_1 * 1024))]));
      half condval_167;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_167 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 166)];
      } else {
        condval_167 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_167 * weight_shared[(((((int)threadIdx.x) & 15) + 3168) - (rw_1 * 1024))]));
      half condval_168;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_168 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 167)];
      } else {
        condval_168 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_168 * weight_shared[(((((int)threadIdx.x) & 15) + 3184) - (rw_1 * 1024))]));
      half condval_169;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_169 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 168)];
      } else {
        condval_169 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_169 * weight_shared[(((((int)threadIdx.x) & 15) + 3200) - (rw_1 * 1024))]));
      half condval_170;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_170 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 169)];
      } else {
        condval_170 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_170 * weight_shared[(((((int)threadIdx.x) & 15) + 3216) - (rw_1 * 1024))]));
      half condval_171;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_171 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 170)];
      } else {
        condval_171 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_171 * weight_shared[(((((int)threadIdx.x) & 15) + 3232) - (rw_1 * 1024))]));
      half condval_172;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_172 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 171)];
      } else {
        condval_172 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_172 * weight_shared[(((((int)threadIdx.x) & 15) + 3248) - (rw_1 * 1024))]));
      half condval_173;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_173 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 172)];
      } else {
        condval_173 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_173 * weight_shared[(((((int)threadIdx.x) & 15) + 3264) - (rw_1 * 1024))]));
      half condval_174;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_174 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 173)];
      } else {
        condval_174 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_174 * weight_shared[(((((int)threadIdx.x) & 15) + 3280) - (rw_1 * 1024))]));
      half condval_175;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_175 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 174)];
      } else {
        condval_175 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_175 * weight_shared[(((((int)threadIdx.x) & 15) + 3296) - (rw_1 * 1024))]));
      half condval_176;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_176 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 175)];
      } else {
        condval_176 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_176 * weight_shared[(((((int)threadIdx.x) & 15) + 3312) - (rw_1 * 1024))]));
      half condval_177;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_177 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 176)];
      } else {
        condval_177 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_177 * weight_shared[(((((int)threadIdx.x) & 15) + 3328) - (rw_1 * 1024))]));
      half condval_178;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_178 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 177)];
      } else {
        condval_178 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_178 * weight_shared[(((((int)threadIdx.x) & 15) + 3344) - (rw_1 * 1024))]));
      half condval_179;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_179 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 178)];
      } else {
        condval_179 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_179 * weight_shared[(((((int)threadIdx.x) & 15) + 3360) - (rw_1 * 1024))]));
      half condval_180;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_180 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 179)];
      } else {
        condval_180 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_180 * weight_shared[(((((int)threadIdx.x) & 15) + 3376) - (rw_1 * 1024))]));
      half condval_181;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_181 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 180)];
      } else {
        condval_181 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_181 * weight_shared[(((((int)threadIdx.x) & 15) + 3392) - (rw_1 * 1024))]));
      half condval_182;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_182 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 181)];
      } else {
        condval_182 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_182 * weight_shared[(((((int)threadIdx.x) & 15) + 3408) - (rw_1 * 1024))]));
      half condval_183;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_183 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 182)];
      } else {
        condval_183 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_183 * weight_shared[(((((int)threadIdx.x) & 15) + 3424) - (rw_1 * 1024))]));
      half condval_184;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_184 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 183)];
      } else {
        condval_184 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_184 * weight_shared[(((((int)threadIdx.x) & 15) + 3440) - (rw_1 * 1024))]));
      half condval_185;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_185 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 184)];
      } else {
        condval_185 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_185 * weight_shared[(((((int)threadIdx.x) & 15) + 3456) - (rw_1 * 1024))]));
      half condval_186;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_186 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 185)];
      } else {
        condval_186 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_186 * weight_shared[(((((int)threadIdx.x) & 15) + 3472) - (rw_1 * 1024))]));
      half condval_187;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_187 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 186)];
      } else {
        condval_187 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_187 * weight_shared[(((((int)threadIdx.x) & 15) + 3488) - (rw_1 * 1024))]));
      half condval_188;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_188 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 187)];
      } else {
        condval_188 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_188 * weight_shared[(((((int)threadIdx.x) & 15) + 3504) - (rw_1 * 1024))]));
      half condval_189;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_189 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 188)];
      } else {
        condval_189 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_189 * weight_shared[(((((int)threadIdx.x) & 15) + 3520) - (rw_1 * 1024))]));
      half condval_190;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_190 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 189)];
      } else {
        condval_190 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_190 * weight_shared[(((((int)threadIdx.x) & 15) + 3536) - (rw_1 * 1024))]));
      half condval_191;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_191 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 190)];
      } else {
        condval_191 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_191 * weight_shared[(((((int)threadIdx.x) & 15) + 3552) - (rw_1 * 1024))]));
      half condval_192;
      if ((((((int)threadIdx.x) & 63) >> 5) == 0)) {
        condval_192 = PadInput_shared[(((((((int)threadIdx.x) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 191)];
      } else {
        condval_192 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_192 * weight_shared[(((((int)threadIdx.x) & 15) + 3568) - (rw_1 * 1024))]));
      half condval_193;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_193 = PadInput_shared[(((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32))];
      } else {
        condval_193 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_193 * weight_shared[(((((int)threadIdx.x) & 15) + 1536) - (rw_1 * 1024))]));
      half condval_194;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_194 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 1)];
      } else {
        condval_194 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_194 * weight_shared[(((((int)threadIdx.x) & 15) + 1552) - (rw_1 * 1024))]));
      half condval_195;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_195 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 2)];
      } else {
        condval_195 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_195 * weight_shared[(((((int)threadIdx.x) & 15) + 1568) - (rw_1 * 1024))]));
      half condval_196;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_196 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 3)];
      } else {
        condval_196 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_196 * weight_shared[(((((int)threadIdx.x) & 15) + 1584) - (rw_1 * 1024))]));
      half condval_197;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_197 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 4)];
      } else {
        condval_197 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_197 * weight_shared[(((((int)threadIdx.x) & 15) + 1600) - (rw_1 * 1024))]));
      half condval_198;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_198 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 5)];
      } else {
        condval_198 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_198 * weight_shared[(((((int)threadIdx.x) & 15) + 1616) - (rw_1 * 1024))]));
      half condval_199;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_199 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 6)];
      } else {
        condval_199 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_199 * weight_shared[(((((int)threadIdx.x) & 15) + 1632) - (rw_1 * 1024))]));
      half condval_200;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_200 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 7)];
      } else {
        condval_200 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_200 * weight_shared[(((((int)threadIdx.x) & 15) + 1648) - (rw_1 * 1024))]));
      half condval_201;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_201 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 8)];
      } else {
        condval_201 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_201 * weight_shared[(((((int)threadIdx.x) & 15) + 1664) - (rw_1 * 1024))]));
      half condval_202;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_202 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 9)];
      } else {
        condval_202 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_202 * weight_shared[(((((int)threadIdx.x) & 15) + 1680) - (rw_1 * 1024))]));
      half condval_203;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_203 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 10)];
      } else {
        condval_203 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_203 * weight_shared[(((((int)threadIdx.x) & 15) + 1696) - (rw_1 * 1024))]));
      half condval_204;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_204 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 11)];
      } else {
        condval_204 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_204 * weight_shared[(((((int)threadIdx.x) & 15) + 1712) - (rw_1 * 1024))]));
      half condval_205;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_205 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 12)];
      } else {
        condval_205 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_205 * weight_shared[(((((int)threadIdx.x) & 15) + 1728) - (rw_1 * 1024))]));
      half condval_206;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_206 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 13)];
      } else {
        condval_206 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_206 * weight_shared[(((((int)threadIdx.x) & 15) + 1744) - (rw_1 * 1024))]));
      half condval_207;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_207 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 14)];
      } else {
        condval_207 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_207 * weight_shared[(((((int)threadIdx.x) & 15) + 1760) - (rw_1 * 1024))]));
      half condval_208;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_208 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 15)];
      } else {
        condval_208 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_208 * weight_shared[(((((int)threadIdx.x) & 15) + 1776) - (rw_1 * 1024))]));
      half condval_209;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_209 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 16)];
      } else {
        condval_209 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_209 * weight_shared[(((((int)threadIdx.x) & 15) + 1792) - (rw_1 * 1024))]));
      half condval_210;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_210 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 17)];
      } else {
        condval_210 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_210 * weight_shared[(((((int)threadIdx.x) & 15) + 1808) - (rw_1 * 1024))]));
      half condval_211;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_211 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 18)];
      } else {
        condval_211 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_211 * weight_shared[(((((int)threadIdx.x) & 15) + 1824) - (rw_1 * 1024))]));
      half condval_212;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_212 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 19)];
      } else {
        condval_212 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_212 * weight_shared[(((((int)threadIdx.x) & 15) + 1840) - (rw_1 * 1024))]));
      half condval_213;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_213 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 20)];
      } else {
        condval_213 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_213 * weight_shared[(((((int)threadIdx.x) & 15) + 1856) - (rw_1 * 1024))]));
      half condval_214;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_214 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 21)];
      } else {
        condval_214 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_214 * weight_shared[(((((int)threadIdx.x) & 15) + 1872) - (rw_1 * 1024))]));
      half condval_215;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_215 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 22)];
      } else {
        condval_215 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_215 * weight_shared[(((((int)threadIdx.x) & 15) + 1888) - (rw_1 * 1024))]));
      half condval_216;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_216 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 23)];
      } else {
        condval_216 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_216 * weight_shared[(((((int)threadIdx.x) & 15) + 1904) - (rw_1 * 1024))]));
      half condval_217;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_217 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 24)];
      } else {
        condval_217 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_217 * weight_shared[(((((int)threadIdx.x) & 15) + 1920) - (rw_1 * 1024))]));
      half condval_218;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_218 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 25)];
      } else {
        condval_218 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_218 * weight_shared[(((((int)threadIdx.x) & 15) + 1936) - (rw_1 * 1024))]));
      half condval_219;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_219 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 26)];
      } else {
        condval_219 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_219 * weight_shared[(((((int)threadIdx.x) & 15) + 1952) - (rw_1 * 1024))]));
      half condval_220;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_220 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 27)];
      } else {
        condval_220 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_220 * weight_shared[(((((int)threadIdx.x) & 15) + 1968) - (rw_1 * 1024))]));
      half condval_221;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_221 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 28)];
      } else {
        condval_221 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_221 * weight_shared[(((((int)threadIdx.x) & 15) + 1984) - (rw_1 * 1024))]));
      half condval_222;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_222 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 29)];
      } else {
        condval_222 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_222 * weight_shared[(((((int)threadIdx.x) & 15) + 2000) - (rw_1 * 1024))]));
      half condval_223;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_223 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 30)];
      } else {
        condval_223 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_223 * weight_shared[(((((int)threadIdx.x) & 15) + 2016) - (rw_1 * 1024))]));
      half condval_224;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_224 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 31)];
      } else {
        condval_224 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (condval_224 * weight_shared[(((((int)threadIdx.x) & 15) + 2032) - (rw_1 * 1024))]));
      half condval_225;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_225 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 32)];
      } else {
        condval_225 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_225 * weight_shared[(((((int)threadIdx.x) & 15) + 1024) - (rw_1 * 1024))]));
      half condval_226;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_226 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 33)];
      } else {
        condval_226 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_226 * weight_shared[(((((int)threadIdx.x) & 15) + 1040) - (rw_1 * 1024))]));
      half condval_227;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_227 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 34)];
      } else {
        condval_227 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_227 * weight_shared[(((((int)threadIdx.x) & 15) + 1056) - (rw_1 * 1024))]));
      half condval_228;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_228 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 35)];
      } else {
        condval_228 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_228 * weight_shared[(((((int)threadIdx.x) & 15) + 1072) - (rw_1 * 1024))]));
      half condval_229;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_229 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 36)];
      } else {
        condval_229 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_229 * weight_shared[(((((int)threadIdx.x) & 15) + 1088) - (rw_1 * 1024))]));
      half condval_230;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_230 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 37)];
      } else {
        condval_230 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_230 * weight_shared[(((((int)threadIdx.x) & 15) + 1104) - (rw_1 * 1024))]));
      half condval_231;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_231 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 38)];
      } else {
        condval_231 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_231 * weight_shared[(((((int)threadIdx.x) & 15) + 1120) - (rw_1 * 1024))]));
      half condval_232;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_232 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 39)];
      } else {
        condval_232 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_232 * weight_shared[(((((int)threadIdx.x) & 15) + 1136) - (rw_1 * 1024))]));
      half condval_233;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_233 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 40)];
      } else {
        condval_233 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_233 * weight_shared[(((((int)threadIdx.x) & 15) + 1152) - (rw_1 * 1024))]));
      half condval_234;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_234 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 41)];
      } else {
        condval_234 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_234 * weight_shared[(((((int)threadIdx.x) & 15) + 1168) - (rw_1 * 1024))]));
      half condval_235;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_235 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 42)];
      } else {
        condval_235 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_235 * weight_shared[(((((int)threadIdx.x) & 15) + 1184) - (rw_1 * 1024))]));
      half condval_236;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_236 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 43)];
      } else {
        condval_236 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_236 * weight_shared[(((((int)threadIdx.x) & 15) + 1200) - (rw_1 * 1024))]));
      half condval_237;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_237 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 44)];
      } else {
        condval_237 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_237 * weight_shared[(((((int)threadIdx.x) & 15) + 1216) - (rw_1 * 1024))]));
      half condval_238;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_238 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 45)];
      } else {
        condval_238 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_238 * weight_shared[(((((int)threadIdx.x) & 15) + 1232) - (rw_1 * 1024))]));
      half condval_239;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_239 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 46)];
      } else {
        condval_239 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_239 * weight_shared[(((((int)threadIdx.x) & 15) + 1248) - (rw_1 * 1024))]));
      half condval_240;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_240 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 47)];
      } else {
        condval_240 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_240 * weight_shared[(((((int)threadIdx.x) & 15) + 1264) - (rw_1 * 1024))]));
      half condval_241;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_241 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 48)];
      } else {
        condval_241 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_241 * weight_shared[(((((int)threadIdx.x) & 15) + 1280) - (rw_1 * 1024))]));
      half condval_242;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_242 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 49)];
      } else {
        condval_242 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_242 * weight_shared[(((((int)threadIdx.x) & 15) + 1296) - (rw_1 * 1024))]));
      half condval_243;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_243 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 50)];
      } else {
        condval_243 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_243 * weight_shared[(((((int)threadIdx.x) & 15) + 1312) - (rw_1 * 1024))]));
      half condval_244;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_244 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 51)];
      } else {
        condval_244 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_244 * weight_shared[(((((int)threadIdx.x) & 15) + 1328) - (rw_1 * 1024))]));
      half condval_245;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_245 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 52)];
      } else {
        condval_245 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_245 * weight_shared[(((((int)threadIdx.x) & 15) + 1344) - (rw_1 * 1024))]));
      half condval_246;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_246 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 53)];
      } else {
        condval_246 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_246 * weight_shared[(((((int)threadIdx.x) & 15) + 1360) - (rw_1 * 1024))]));
      half condval_247;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_247 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 54)];
      } else {
        condval_247 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_247 * weight_shared[(((((int)threadIdx.x) & 15) + 1376) - (rw_1 * 1024))]));
      half condval_248;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_248 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 55)];
      } else {
        condval_248 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_248 * weight_shared[(((((int)threadIdx.x) & 15) + 1392) - (rw_1 * 1024))]));
      half condval_249;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_249 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 56)];
      } else {
        condval_249 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_249 * weight_shared[(((((int)threadIdx.x) & 15) + 1408) - (rw_1 * 1024))]));
      half condval_250;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_250 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 57)];
      } else {
        condval_250 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_250 * weight_shared[(((((int)threadIdx.x) & 15) + 1424) - (rw_1 * 1024))]));
      half condval_251;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_251 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 58)];
      } else {
        condval_251 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_251 * weight_shared[(((((int)threadIdx.x) & 15) + 1440) - (rw_1 * 1024))]));
      half condval_252;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_252 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 59)];
      } else {
        condval_252 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_252 * weight_shared[(((((int)threadIdx.x) & 15) + 1456) - (rw_1 * 1024))]));
      half condval_253;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_253 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 60)];
      } else {
        condval_253 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_253 * weight_shared[(((((int)threadIdx.x) & 15) + 1472) - (rw_1 * 1024))]));
      half condval_254;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_254 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 61)];
      } else {
        condval_254 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_254 * weight_shared[(((((int)threadIdx.x) & 15) + 1488) - (rw_1 * 1024))]));
      half condval_255;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_255 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 62)];
      } else {
        condval_255 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_255 * weight_shared[(((((int)threadIdx.x) & 15) + 1504) - (rw_1 * 1024))]));
      half condval_256;
      if (((((((int)threadIdx.x) >> 5) + 1) % 2) == 0)) {
        condval_256 = PadInput_shared[((((((((int)threadIdx.x) + 96) >> 6) * 128) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (rw_1 * 32)) + 63)];
      } else {
        condval_256 = __float2half_rn(0.000000e+00f);
      }
      conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (condval_256 * weight_shared[(((((int)threadIdx.x) & 15) + 1520) - (rw_1 * 1024))]));
    }
  }
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + (((((int)blockIdx.x) & 31) >> 4) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[((((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + (((((int)blockIdx.x) & 31) >> 4) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 256)] = conv2d_transpose_nhwc_local[1];
}

