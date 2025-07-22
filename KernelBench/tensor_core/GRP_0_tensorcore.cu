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
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight) {
  half conv2d_nhwc_local[4];
  __shared__ half PadInput_shared[1296];
  __shared__ half weight_shared[2304];
  conv2d_nhwc_local[0] = __float2half_rn(0.000000e+00f);
  conv2d_nhwc_local[2] = __float2half_rn(0.000000e+00f);
  conv2d_nhwc_local[1] = __float2half_rn(0.000000e+00f);
  conv2d_nhwc_local[3] = __float2half_rn(0.000000e+00f);
  half condval;
  if (((56 <= ((int)blockIdx.x)) && (1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((int)threadIdx.x) >> 3))))) {
    condval = inputs[(((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) * 2)] = condval;
  half condval_1;
  if (((56 <= ((int)blockIdx.x)) && (1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((int)threadIdx.x) >> 3))))) {
    condval_1 = inputs[(((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_1 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 1)] = condval_1;
  half condval_2;
  if (((1 <= (((((int)blockIdx.x) / 56) * 8) + ((((int)threadIdx.x) + 64) / 72))) && (1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 8) % 9))))) {
    condval_2 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 64) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 8) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval_2 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 128)] = condval_2;
  half condval_3;
  if (((1 <= (((((int)blockIdx.x) / 56) * 8) + ((((int)threadIdx.x) + 64) / 72))) && (1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 8) % 9))))) {
    condval_3 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 64) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 8) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_3 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 129)] = condval_3;
  half condval_4;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 7) % 9)))) {
    condval_4 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 128) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 7) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval_4 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 256)] = condval_4;
  half condval_5;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 7) % 9)))) {
    condval_5 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 128) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 7) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_5 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 257)] = condval_5;
  half condval_6;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 6) % 9)))) {
    condval_6 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 192) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 6) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval_6 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 384)] = condval_6;
  half condval_7;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 6) % 9)))) {
    condval_7 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 192) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 6) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_7 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 385)] = condval_7;
  half condval_8;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 5) % 9)))) {
    condval_8 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 256) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 5) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval_8 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 512)] = condval_8;
  half condval_9;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 5) % 9)))) {
    condval_9 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 256) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 5) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_9 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 513)] = condval_9;
  half condval_10;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 4) % 9)))) {
    condval_10 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 320) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval_10 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 640)] = condval_10;
  half condval_11;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 4) % 9)))) {
    condval_11 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 320) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 4) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_11 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 641)] = condval_11;
  half condval_12;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 3) % 9)))) {
    condval_12 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 384) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval_12 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 768)] = condval_12;
  half condval_13;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 3) % 9)))) {
    condval_13 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 384) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 3) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_13 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 769)] = condval_13;
  half condval_14;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 2) % 9)))) {
    condval_14 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 448) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3648)];
  } else {
    condval_14 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 896)] = condval_14;
  half condval_15;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((((int)threadIdx.x) >> 3) + 2) % 9)))) {
    condval_15 = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 448) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((((int)threadIdx.x) >> 3) + 2) % 9) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3647)];
  } else {
    condval_15 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 897)] = condval_15;
  PadInput_shared[((((int)threadIdx.x) * 2) + 1024)] = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 512) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3584)];
  PadInput_shared[((((int)threadIdx.x) * 2) + 1025)] = inputs[((((((((((int)blockIdx.x) / 56) * 28672) + (((((int)threadIdx.x) + 512) / 72) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) - 3583)];
  half condval_16;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((int)threadIdx.x) >> 3)))) {
    condval_16 = inputs[(((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 25024)];
  } else {
    condval_16 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 1152)] = condval_16;
  half condval_17;
  if ((1 <= ((((((int)blockIdx.x) % 56) >> 3) * 8) + (((int)threadIdx.x) >> 3)))) {
    condval_17 = inputs[(((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 25025)];
  } else {
    condval_17 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((((int)threadIdx.x) * 2) + 1153)] = condval_17;
  if (((int)threadIdx.x) < 8) {
    PadInput_shared[((((int)threadIdx.x) * 2) + 1280)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (((int)threadIdx.x) * 2)) + 25536)];
    PadInput_shared[((((int)threadIdx.x) * 2) + 1281)] = inputs[((((((((int)blockIdx.x) / 56) * 28672) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 16)) + (((int)threadIdx.x) * 2)) + 25537)];
  }
  *(uint4*)(weight_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(weight + ((((((int)threadIdx.x) >> 1) * 128) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
  *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 512)) = *(uint4*)(weight + (((((((int)threadIdx.x) >> 1) * 128) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 4096));
  *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4*)(weight + (((((((int)threadIdx.x) >> 1) * 128) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 8192));
  *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 1536)) = *(uint4*)(weight + (((((((int)threadIdx.x) >> 1) * 128) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 12288));
  if (((int)threadIdx.x) < 32) {
    *(uint4*)(weight_shared + ((((int)threadIdx.x) * 8) + 2048)) = *(uint4*)(weight + (((((((int)threadIdx.x) >> 1) * 128) + ((((int)blockIdx.x) & 7) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 16384));
  }
  __syncthreads();
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) >> 4) * 288)] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 64)] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 1)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 65)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 2)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 66)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 3)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 67)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 4)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 68)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 5)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 69)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 6)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 70)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 7)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 71)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 8)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 72)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 9)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 73)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 10)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 74)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 11)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 75)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 12)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 76)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 13)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 77)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 14)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 78)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 15)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 79)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 144)] * weight_shared[((((int)threadIdx.x) & 15) + 768)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 208)] * weight_shared[((((int)threadIdx.x) & 15) + 768)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 145)] * weight_shared[((((int)threadIdx.x) & 15) + 784)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 209)] * weight_shared[((((int)threadIdx.x) & 15) + 784)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 146)] * weight_shared[((((int)threadIdx.x) & 15) + 800)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 210)] * weight_shared[((((int)threadIdx.x) & 15) + 800)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 147)] * weight_shared[((((int)threadIdx.x) & 15) + 816)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 211)] * weight_shared[((((int)threadIdx.x) & 15) + 816)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 148)] * weight_shared[((((int)threadIdx.x) & 15) + 832)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 212)] * weight_shared[((((int)threadIdx.x) & 15) + 832)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 149)] * weight_shared[((((int)threadIdx.x) & 15) + 848)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 213)] * weight_shared[((((int)threadIdx.x) & 15) + 848)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 150)] * weight_shared[((((int)threadIdx.x) & 15) + 864)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 214)] * weight_shared[((((int)threadIdx.x) & 15) + 864)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 151)] * weight_shared[((((int)threadIdx.x) & 15) + 880)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 215)] * weight_shared[((((int)threadIdx.x) & 15) + 880)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 152)] * weight_shared[((((int)threadIdx.x) & 15) + 896)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 216)] * weight_shared[((((int)threadIdx.x) & 15) + 896)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 153)] * weight_shared[((((int)threadIdx.x) & 15) + 912)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 217)] * weight_shared[((((int)threadIdx.x) & 15) + 912)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 154)] * weight_shared[((((int)threadIdx.x) & 15) + 928)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 218)] * weight_shared[((((int)threadIdx.x) & 15) + 928)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 155)] * weight_shared[((((int)threadIdx.x) & 15) + 944)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 219)] * weight_shared[((((int)threadIdx.x) & 15) + 944)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 156)] * weight_shared[((((int)threadIdx.x) & 15) + 960)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 220)] * weight_shared[((((int)threadIdx.x) & 15) + 960)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 157)] * weight_shared[((((int)threadIdx.x) & 15) + 976)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 221)] * weight_shared[((((int)threadIdx.x) & 15) + 976)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 158)] * weight_shared[((((int)threadIdx.x) & 15) + 992)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 222)] * weight_shared[((((int)threadIdx.x) & 15) + 992)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 159)] * weight_shared[((((int)threadIdx.x) & 15) + 1008)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 223)] * weight_shared[((((int)threadIdx.x) & 15) + 1008)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 288)] * weight_shared[((((int)threadIdx.x) & 15) + 1536)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 352)] * weight_shared[((((int)threadIdx.x) & 15) + 1536)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 289)] * weight_shared[((((int)threadIdx.x) & 15) + 1552)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 353)] * weight_shared[((((int)threadIdx.x) & 15) + 1552)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 290)] * weight_shared[((((int)threadIdx.x) & 15) + 1568)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 354)] * weight_shared[((((int)threadIdx.x) & 15) + 1568)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 291)] * weight_shared[((((int)threadIdx.x) & 15) + 1584)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 355)] * weight_shared[((((int)threadIdx.x) & 15) + 1584)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 292)] * weight_shared[((((int)threadIdx.x) & 15) + 1600)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 356)] * weight_shared[((((int)threadIdx.x) & 15) + 1600)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 293)] * weight_shared[((((int)threadIdx.x) & 15) + 1616)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 357)] * weight_shared[((((int)threadIdx.x) & 15) + 1616)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 294)] * weight_shared[((((int)threadIdx.x) & 15) + 1632)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 358)] * weight_shared[((((int)threadIdx.x) & 15) + 1632)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 295)] * weight_shared[((((int)threadIdx.x) & 15) + 1648)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 359)] * weight_shared[((((int)threadIdx.x) & 15) + 1648)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 296)] * weight_shared[((((int)threadIdx.x) & 15) + 1664)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 360)] * weight_shared[((((int)threadIdx.x) & 15) + 1664)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 297)] * weight_shared[((((int)threadIdx.x) & 15) + 1680)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 361)] * weight_shared[((((int)threadIdx.x) & 15) + 1680)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 298)] * weight_shared[((((int)threadIdx.x) & 15) + 1696)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 362)] * weight_shared[((((int)threadIdx.x) & 15) + 1696)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 299)] * weight_shared[((((int)threadIdx.x) & 15) + 1712)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 363)] * weight_shared[((((int)threadIdx.x) & 15) + 1712)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 300)] * weight_shared[((((int)threadIdx.x) & 15) + 1728)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 364)] * weight_shared[((((int)threadIdx.x) & 15) + 1728)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 301)] * weight_shared[((((int)threadIdx.x) & 15) + 1744)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 365)] * weight_shared[((((int)threadIdx.x) & 15) + 1744)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 302)] * weight_shared[((((int)threadIdx.x) & 15) + 1760)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 366)] * weight_shared[((((int)threadIdx.x) & 15) + 1760)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 303)] * weight_shared[((((int)threadIdx.x) & 15) + 1776)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 367)] * weight_shared[((((int)threadIdx.x) & 15) + 1776)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 32)] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 96)] * weight_shared[(((int)threadIdx.x) & 15)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 33)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 97)] * weight_shared[((((int)threadIdx.x) & 15) + 16)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 34)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 98)] * weight_shared[((((int)threadIdx.x) & 15) + 32)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 35)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 99)] * weight_shared[((((int)threadIdx.x) & 15) + 48)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 36)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 100)] * weight_shared[((((int)threadIdx.x) & 15) + 64)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 37)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 101)] * weight_shared[((((int)threadIdx.x) & 15) + 80)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 38)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 102)] * weight_shared[((((int)threadIdx.x) & 15) + 96)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 39)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 103)] * weight_shared[((((int)threadIdx.x) & 15) + 112)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 40)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 104)] * weight_shared[((((int)threadIdx.x) & 15) + 128)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 41)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 105)] * weight_shared[((((int)threadIdx.x) & 15) + 144)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 42)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 106)] * weight_shared[((((int)threadIdx.x) & 15) + 160)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 43)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 107)] * weight_shared[((((int)threadIdx.x) & 15) + 176)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 44)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 108)] * weight_shared[((((int)threadIdx.x) & 15) + 192)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 45)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 109)] * weight_shared[((((int)threadIdx.x) & 15) + 208)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 46)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 110)] * weight_shared[((((int)threadIdx.x) & 15) + 224)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 47)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 111)] * weight_shared[((((int)threadIdx.x) & 15) + 240)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 176)] * weight_shared[((((int)threadIdx.x) & 15) + 768)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 240)] * weight_shared[((((int)threadIdx.x) & 15) + 768)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 177)] * weight_shared[((((int)threadIdx.x) & 15) + 784)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 241)] * weight_shared[((((int)threadIdx.x) & 15) + 784)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 178)] * weight_shared[((((int)threadIdx.x) & 15) + 800)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 242)] * weight_shared[((((int)threadIdx.x) & 15) + 800)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 179)] * weight_shared[((((int)threadIdx.x) & 15) + 816)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 243)] * weight_shared[((((int)threadIdx.x) & 15) + 816)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 180)] * weight_shared[((((int)threadIdx.x) & 15) + 832)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 244)] * weight_shared[((((int)threadIdx.x) & 15) + 832)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 181)] * weight_shared[((((int)threadIdx.x) & 15) + 848)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 245)] * weight_shared[((((int)threadIdx.x) & 15) + 848)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 182)] * weight_shared[((((int)threadIdx.x) & 15) + 864)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 246)] * weight_shared[((((int)threadIdx.x) & 15) + 864)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 183)] * weight_shared[((((int)threadIdx.x) & 15) + 880)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 247)] * weight_shared[((((int)threadIdx.x) & 15) + 880)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 184)] * weight_shared[((((int)threadIdx.x) & 15) + 896)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 248)] * weight_shared[((((int)threadIdx.x) & 15) + 896)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 185)] * weight_shared[((((int)threadIdx.x) & 15) + 912)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 249)] * weight_shared[((((int)threadIdx.x) & 15) + 912)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 186)] * weight_shared[((((int)threadIdx.x) & 15) + 928)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 250)] * weight_shared[((((int)threadIdx.x) & 15) + 928)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 187)] * weight_shared[((((int)threadIdx.x) & 15) + 944)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 251)] * weight_shared[((((int)threadIdx.x) & 15) + 944)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 188)] * weight_shared[((((int)threadIdx.x) & 15) + 960)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 252)] * weight_shared[((((int)threadIdx.x) & 15) + 960)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 189)] * weight_shared[((((int)threadIdx.x) & 15) + 976)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 253)] * weight_shared[((((int)threadIdx.x) & 15) + 976)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 190)] * weight_shared[((((int)threadIdx.x) & 15) + 992)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 254)] * weight_shared[((((int)threadIdx.x) & 15) + 992)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 191)] * weight_shared[((((int)threadIdx.x) & 15) + 1008)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 255)] * weight_shared[((((int)threadIdx.x) & 15) + 1008)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 320)] * weight_shared[((((int)threadIdx.x) & 15) + 1536)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 384)] * weight_shared[((((int)threadIdx.x) & 15) + 1536)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 321)] * weight_shared[((((int)threadIdx.x) & 15) + 1552)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 385)] * weight_shared[((((int)threadIdx.x) & 15) + 1552)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 322)] * weight_shared[((((int)threadIdx.x) & 15) + 1568)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 386)] * weight_shared[((((int)threadIdx.x) & 15) + 1568)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 323)] * weight_shared[((((int)threadIdx.x) & 15) + 1584)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 387)] * weight_shared[((((int)threadIdx.x) & 15) + 1584)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 324)] * weight_shared[((((int)threadIdx.x) & 15) + 1600)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 388)] * weight_shared[((((int)threadIdx.x) & 15) + 1600)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 325)] * weight_shared[((((int)threadIdx.x) & 15) + 1616)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 389)] * weight_shared[((((int)threadIdx.x) & 15) + 1616)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 326)] * weight_shared[((((int)threadIdx.x) & 15) + 1632)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 390)] * weight_shared[((((int)threadIdx.x) & 15) + 1632)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 327)] * weight_shared[((((int)threadIdx.x) & 15) + 1648)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 391)] * weight_shared[((((int)threadIdx.x) & 15) + 1648)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 328)] * weight_shared[((((int)threadIdx.x) & 15) + 1664)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 392)] * weight_shared[((((int)threadIdx.x) & 15) + 1664)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 329)] * weight_shared[((((int)threadIdx.x) & 15) + 1680)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 393)] * weight_shared[((((int)threadIdx.x) & 15) + 1680)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 330)] * weight_shared[((((int)threadIdx.x) & 15) + 1696)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 394)] * weight_shared[((((int)threadIdx.x) & 15) + 1696)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 331)] * weight_shared[((((int)threadIdx.x) & 15) + 1712)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 395)] * weight_shared[((((int)threadIdx.x) & 15) + 1712)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 332)] * weight_shared[((((int)threadIdx.x) & 15) + 1728)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 396)] * weight_shared[((((int)threadIdx.x) & 15) + 1728)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 333)] * weight_shared[((((int)threadIdx.x) & 15) + 1744)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 397)] * weight_shared[((((int)threadIdx.x) & 15) + 1744)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 334)] * weight_shared[((((int)threadIdx.x) & 15) + 1760)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 398)] * weight_shared[((((int)threadIdx.x) & 15) + 1760)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 335)] * weight_shared[((((int)threadIdx.x) & 15) + 1776)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 399)] * weight_shared[((((int)threadIdx.x) & 15) + 1776)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 16)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 80)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 17)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 81)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 18)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 82)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 19)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 83)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 20)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 84)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 21)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 85)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 22)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 86)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 23)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 87)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 24)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 88)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 25)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 89)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 26)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 90)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 27)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 91)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 28)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 92)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 29)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 93)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 30)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 94)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 31)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 95)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 160)] * weight_shared[((((int)threadIdx.x) & 15) + 1024)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 224)] * weight_shared[((((int)threadIdx.x) & 15) + 1024)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 161)] * weight_shared[((((int)threadIdx.x) & 15) + 1040)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 225)] * weight_shared[((((int)threadIdx.x) & 15) + 1040)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 162)] * weight_shared[((((int)threadIdx.x) & 15) + 1056)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 226)] * weight_shared[((((int)threadIdx.x) & 15) + 1056)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 163)] * weight_shared[((((int)threadIdx.x) & 15) + 1072)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 227)] * weight_shared[((((int)threadIdx.x) & 15) + 1072)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 164)] * weight_shared[((((int)threadIdx.x) & 15) + 1088)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 228)] * weight_shared[((((int)threadIdx.x) & 15) + 1088)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 165)] * weight_shared[((((int)threadIdx.x) & 15) + 1104)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 229)] * weight_shared[((((int)threadIdx.x) & 15) + 1104)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 166)] * weight_shared[((((int)threadIdx.x) & 15) + 1120)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 230)] * weight_shared[((((int)threadIdx.x) & 15) + 1120)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 167)] * weight_shared[((((int)threadIdx.x) & 15) + 1136)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 231)] * weight_shared[((((int)threadIdx.x) & 15) + 1136)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 168)] * weight_shared[((((int)threadIdx.x) & 15) + 1152)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 232)] * weight_shared[((((int)threadIdx.x) & 15) + 1152)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 169)] * weight_shared[((((int)threadIdx.x) & 15) + 1168)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 233)] * weight_shared[((((int)threadIdx.x) & 15) + 1168)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 170)] * weight_shared[((((int)threadIdx.x) & 15) + 1184)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 234)] * weight_shared[((((int)threadIdx.x) & 15) + 1184)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 171)] * weight_shared[((((int)threadIdx.x) & 15) + 1200)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 235)] * weight_shared[((((int)threadIdx.x) & 15) + 1200)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 172)] * weight_shared[((((int)threadIdx.x) & 15) + 1216)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 236)] * weight_shared[((((int)threadIdx.x) & 15) + 1216)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 173)] * weight_shared[((((int)threadIdx.x) & 15) + 1232)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 237)] * weight_shared[((((int)threadIdx.x) & 15) + 1232)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 174)] * weight_shared[((((int)threadIdx.x) & 15) + 1248)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 238)] * weight_shared[((((int)threadIdx.x) & 15) + 1248)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 175)] * weight_shared[((((int)threadIdx.x) & 15) + 1264)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 239)] * weight_shared[((((int)threadIdx.x) & 15) + 1264)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 304)] * weight_shared[((((int)threadIdx.x) & 15) + 1792)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 368)] * weight_shared[((((int)threadIdx.x) & 15) + 1792)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 305)] * weight_shared[((((int)threadIdx.x) & 15) + 1808)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 369)] * weight_shared[((((int)threadIdx.x) & 15) + 1808)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 306)] * weight_shared[((((int)threadIdx.x) & 15) + 1824)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 370)] * weight_shared[((((int)threadIdx.x) & 15) + 1824)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 307)] * weight_shared[((((int)threadIdx.x) & 15) + 1840)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 371)] * weight_shared[((((int)threadIdx.x) & 15) + 1840)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 308)] * weight_shared[((((int)threadIdx.x) & 15) + 1856)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 372)] * weight_shared[((((int)threadIdx.x) & 15) + 1856)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 309)] * weight_shared[((((int)threadIdx.x) & 15) + 1872)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 373)] * weight_shared[((((int)threadIdx.x) & 15) + 1872)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 310)] * weight_shared[((((int)threadIdx.x) & 15) + 1888)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 374)] * weight_shared[((((int)threadIdx.x) & 15) + 1888)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 311)] * weight_shared[((((int)threadIdx.x) & 15) + 1904)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 375)] * weight_shared[((((int)threadIdx.x) & 15) + 1904)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 312)] * weight_shared[((((int)threadIdx.x) & 15) + 1920)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 376)] * weight_shared[((((int)threadIdx.x) & 15) + 1920)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 313)] * weight_shared[((((int)threadIdx.x) & 15) + 1936)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 377)] * weight_shared[((((int)threadIdx.x) & 15) + 1936)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 314)] * weight_shared[((((int)threadIdx.x) & 15) + 1952)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 378)] * weight_shared[((((int)threadIdx.x) & 15) + 1952)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 315)] * weight_shared[((((int)threadIdx.x) & 15) + 1968)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 379)] * weight_shared[((((int)threadIdx.x) & 15) + 1968)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 316)] * weight_shared[((((int)threadIdx.x) & 15) + 1984)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 380)] * weight_shared[((((int)threadIdx.x) & 15) + 1984)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 317)] * weight_shared[((((int)threadIdx.x) & 15) + 2000)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 381)] * weight_shared[((((int)threadIdx.x) & 15) + 2000)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 318)] * weight_shared[((((int)threadIdx.x) & 15) + 2016)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 382)] * weight_shared[((((int)threadIdx.x) & 15) + 2016)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 319)] * weight_shared[((((int)threadIdx.x) & 15) + 2032)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 383)] * weight_shared[((((int)threadIdx.x) & 15) + 2032)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 48)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 112)] * weight_shared[((((int)threadIdx.x) & 15) + 256)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 49)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 113)] * weight_shared[((((int)threadIdx.x) & 15) + 272)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 50)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 114)] * weight_shared[((((int)threadIdx.x) & 15) + 288)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 51)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 115)] * weight_shared[((((int)threadIdx.x) & 15) + 304)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 52)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 116)] * weight_shared[((((int)threadIdx.x) & 15) + 320)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 53)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 117)] * weight_shared[((((int)threadIdx.x) & 15) + 336)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 54)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 118)] * weight_shared[((((int)threadIdx.x) & 15) + 352)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 55)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 119)] * weight_shared[((((int)threadIdx.x) & 15) + 368)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 56)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 120)] * weight_shared[((((int)threadIdx.x) & 15) + 384)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 57)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 121)] * weight_shared[((((int)threadIdx.x) & 15) + 400)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 58)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 122)] * weight_shared[((((int)threadIdx.x) & 15) + 416)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 59)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 123)] * weight_shared[((((int)threadIdx.x) & 15) + 432)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 60)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 124)] * weight_shared[((((int)threadIdx.x) & 15) + 448)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 61)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 125)] * weight_shared[((((int)threadIdx.x) & 15) + 464)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 62)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 126)] * weight_shared[((((int)threadIdx.x) & 15) + 480)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 63)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 127)] * weight_shared[((((int)threadIdx.x) & 15) + 496)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 192)] * weight_shared[((((int)threadIdx.x) & 15) + 1024)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 256)] * weight_shared[((((int)threadIdx.x) & 15) + 1024)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 193)] * weight_shared[((((int)threadIdx.x) & 15) + 1040)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 257)] * weight_shared[((((int)threadIdx.x) & 15) + 1040)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 194)] * weight_shared[((((int)threadIdx.x) & 15) + 1056)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 258)] * weight_shared[((((int)threadIdx.x) & 15) + 1056)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 195)] * weight_shared[((((int)threadIdx.x) & 15) + 1072)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 259)] * weight_shared[((((int)threadIdx.x) & 15) + 1072)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 196)] * weight_shared[((((int)threadIdx.x) & 15) + 1088)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 260)] * weight_shared[((((int)threadIdx.x) & 15) + 1088)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 197)] * weight_shared[((((int)threadIdx.x) & 15) + 1104)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 261)] * weight_shared[((((int)threadIdx.x) & 15) + 1104)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 198)] * weight_shared[((((int)threadIdx.x) & 15) + 1120)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 262)] * weight_shared[((((int)threadIdx.x) & 15) + 1120)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 199)] * weight_shared[((((int)threadIdx.x) & 15) + 1136)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 263)] * weight_shared[((((int)threadIdx.x) & 15) + 1136)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 200)] * weight_shared[((((int)threadIdx.x) & 15) + 1152)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 264)] * weight_shared[((((int)threadIdx.x) & 15) + 1152)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 201)] * weight_shared[((((int)threadIdx.x) & 15) + 1168)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 265)] * weight_shared[((((int)threadIdx.x) & 15) + 1168)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 202)] * weight_shared[((((int)threadIdx.x) & 15) + 1184)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 266)] * weight_shared[((((int)threadIdx.x) & 15) + 1184)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 203)] * weight_shared[((((int)threadIdx.x) & 15) + 1200)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 267)] * weight_shared[((((int)threadIdx.x) & 15) + 1200)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 204)] * weight_shared[((((int)threadIdx.x) & 15) + 1216)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 268)] * weight_shared[((((int)threadIdx.x) & 15) + 1216)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 205)] * weight_shared[((((int)threadIdx.x) & 15) + 1232)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 269)] * weight_shared[((((int)threadIdx.x) & 15) + 1232)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 206)] * weight_shared[((((int)threadIdx.x) & 15) + 1248)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 270)] * weight_shared[((((int)threadIdx.x) & 15) + 1248)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 207)] * weight_shared[((((int)threadIdx.x) & 15) + 1264)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 271)] * weight_shared[((((int)threadIdx.x) & 15) + 1264)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 336)] * weight_shared[((((int)threadIdx.x) & 15) + 1792)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 400)] * weight_shared[((((int)threadIdx.x) & 15) + 1792)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 337)] * weight_shared[((((int)threadIdx.x) & 15) + 1808)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 401)] * weight_shared[((((int)threadIdx.x) & 15) + 1808)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 338)] * weight_shared[((((int)threadIdx.x) & 15) + 1824)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 402)] * weight_shared[((((int)threadIdx.x) & 15) + 1824)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 339)] * weight_shared[((((int)threadIdx.x) & 15) + 1840)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 403)] * weight_shared[((((int)threadIdx.x) & 15) + 1840)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 340)] * weight_shared[((((int)threadIdx.x) & 15) + 1856)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 404)] * weight_shared[((((int)threadIdx.x) & 15) + 1856)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 341)] * weight_shared[((((int)threadIdx.x) & 15) + 1872)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 405)] * weight_shared[((((int)threadIdx.x) & 15) + 1872)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 342)] * weight_shared[((((int)threadIdx.x) & 15) + 1888)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 406)] * weight_shared[((((int)threadIdx.x) & 15) + 1888)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 343)] * weight_shared[((((int)threadIdx.x) & 15) + 1904)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 407)] * weight_shared[((((int)threadIdx.x) & 15) + 1904)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 344)] * weight_shared[((((int)threadIdx.x) & 15) + 1920)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 408)] * weight_shared[((((int)threadIdx.x) & 15) + 1920)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 345)] * weight_shared[((((int)threadIdx.x) & 15) + 1936)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 409)] * weight_shared[((((int)threadIdx.x) & 15) + 1936)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 346)] * weight_shared[((((int)threadIdx.x) & 15) + 1952)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 410)] * weight_shared[((((int)threadIdx.x) & 15) + 1952)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 347)] * weight_shared[((((int)threadIdx.x) & 15) + 1968)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 411)] * weight_shared[((((int)threadIdx.x) & 15) + 1968)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 348)] * weight_shared[((((int)threadIdx.x) & 15) + 1984)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 412)] * weight_shared[((((int)threadIdx.x) & 15) + 1984)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 349)] * weight_shared[((((int)threadIdx.x) & 15) + 2000)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 413)] * weight_shared[((((int)threadIdx.x) & 15) + 2000)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 350)] * weight_shared[((((int)threadIdx.x) & 15) + 2016)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 414)] * weight_shared[((((int)threadIdx.x) & 15) + 2016)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 351)] * weight_shared[((((int)threadIdx.x) & 15) + 2032)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 415)] * weight_shared[((((int)threadIdx.x) & 15) + 2032)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 32)] * weight_shared[((((int)threadIdx.x) & 15) + 512)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 96)] * weight_shared[((((int)threadIdx.x) & 15) + 512)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 33)] * weight_shared[((((int)threadIdx.x) & 15) + 528)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 97)] * weight_shared[((((int)threadIdx.x) & 15) + 528)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 34)] * weight_shared[((((int)threadIdx.x) & 15) + 544)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 98)] * weight_shared[((((int)threadIdx.x) & 15) + 544)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 35)] * weight_shared[((((int)threadIdx.x) & 15) + 560)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 99)] * weight_shared[((((int)threadIdx.x) & 15) + 560)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 36)] * weight_shared[((((int)threadIdx.x) & 15) + 576)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 100)] * weight_shared[((((int)threadIdx.x) & 15) + 576)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 37)] * weight_shared[((((int)threadIdx.x) & 15) + 592)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 101)] * weight_shared[((((int)threadIdx.x) & 15) + 592)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 38)] * weight_shared[((((int)threadIdx.x) & 15) + 608)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 102)] * weight_shared[((((int)threadIdx.x) & 15) + 608)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 39)] * weight_shared[((((int)threadIdx.x) & 15) + 624)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 103)] * weight_shared[((((int)threadIdx.x) & 15) + 624)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 40)] * weight_shared[((((int)threadIdx.x) & 15) + 640)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 104)] * weight_shared[((((int)threadIdx.x) & 15) + 640)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 41)] * weight_shared[((((int)threadIdx.x) & 15) + 656)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 105)] * weight_shared[((((int)threadIdx.x) & 15) + 656)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 42)] * weight_shared[((((int)threadIdx.x) & 15) + 672)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 106)] * weight_shared[((((int)threadIdx.x) & 15) + 672)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 43)] * weight_shared[((((int)threadIdx.x) & 15) + 688)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 107)] * weight_shared[((((int)threadIdx.x) & 15) + 688)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 44)] * weight_shared[((((int)threadIdx.x) & 15) + 704)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 108)] * weight_shared[((((int)threadIdx.x) & 15) + 704)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 45)] * weight_shared[((((int)threadIdx.x) & 15) + 720)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 109)] * weight_shared[((((int)threadIdx.x) & 15) + 720)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 46)] * weight_shared[((((int)threadIdx.x) & 15) + 736)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 110)] * weight_shared[((((int)threadIdx.x) & 15) + 736)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 47)] * weight_shared[((((int)threadIdx.x) & 15) + 752)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 111)] * weight_shared[((((int)threadIdx.x) & 15) + 752)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 176)] * weight_shared[((((int)threadIdx.x) & 15) + 1280)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 240)] * weight_shared[((((int)threadIdx.x) & 15) + 1280)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 177)] * weight_shared[((((int)threadIdx.x) & 15) + 1296)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 241)] * weight_shared[((((int)threadIdx.x) & 15) + 1296)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 178)] * weight_shared[((((int)threadIdx.x) & 15) + 1312)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 242)] * weight_shared[((((int)threadIdx.x) & 15) + 1312)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 179)] * weight_shared[((((int)threadIdx.x) & 15) + 1328)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 243)] * weight_shared[((((int)threadIdx.x) & 15) + 1328)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 180)] * weight_shared[((((int)threadIdx.x) & 15) + 1344)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 244)] * weight_shared[((((int)threadIdx.x) & 15) + 1344)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 181)] * weight_shared[((((int)threadIdx.x) & 15) + 1360)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 245)] * weight_shared[((((int)threadIdx.x) & 15) + 1360)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 182)] * weight_shared[((((int)threadIdx.x) & 15) + 1376)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 246)] * weight_shared[((((int)threadIdx.x) & 15) + 1376)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 183)] * weight_shared[((((int)threadIdx.x) & 15) + 1392)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 247)] * weight_shared[((((int)threadIdx.x) & 15) + 1392)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 184)] * weight_shared[((((int)threadIdx.x) & 15) + 1408)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 248)] * weight_shared[((((int)threadIdx.x) & 15) + 1408)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 185)] * weight_shared[((((int)threadIdx.x) & 15) + 1424)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 249)] * weight_shared[((((int)threadIdx.x) & 15) + 1424)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 186)] * weight_shared[((((int)threadIdx.x) & 15) + 1440)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 250)] * weight_shared[((((int)threadIdx.x) & 15) + 1440)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 187)] * weight_shared[((((int)threadIdx.x) & 15) + 1456)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 251)] * weight_shared[((((int)threadIdx.x) & 15) + 1456)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 188)] * weight_shared[((((int)threadIdx.x) & 15) + 1472)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 252)] * weight_shared[((((int)threadIdx.x) & 15) + 1472)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 189)] * weight_shared[((((int)threadIdx.x) & 15) + 1488)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 253)] * weight_shared[((((int)threadIdx.x) & 15) + 1488)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 190)] * weight_shared[((((int)threadIdx.x) & 15) + 1504)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 254)] * weight_shared[((((int)threadIdx.x) & 15) + 1504)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 191)] * weight_shared[((((int)threadIdx.x) & 15) + 1520)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 255)] * weight_shared[((((int)threadIdx.x) & 15) + 1520)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 320)] * weight_shared[((((int)threadIdx.x) & 15) + 2048)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 384)] * weight_shared[((((int)threadIdx.x) & 15) + 2048)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 321)] * weight_shared[((((int)threadIdx.x) & 15) + 2064)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 385)] * weight_shared[((((int)threadIdx.x) & 15) + 2064)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 322)] * weight_shared[((((int)threadIdx.x) & 15) + 2080)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 386)] * weight_shared[((((int)threadIdx.x) & 15) + 2080)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 323)] * weight_shared[((((int)threadIdx.x) & 15) + 2096)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 387)] * weight_shared[((((int)threadIdx.x) & 15) + 2096)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 324)] * weight_shared[((((int)threadIdx.x) & 15) + 2112)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 388)] * weight_shared[((((int)threadIdx.x) & 15) + 2112)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 325)] * weight_shared[((((int)threadIdx.x) & 15) + 2128)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 389)] * weight_shared[((((int)threadIdx.x) & 15) + 2128)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 326)] * weight_shared[((((int)threadIdx.x) & 15) + 2144)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 390)] * weight_shared[((((int)threadIdx.x) & 15) + 2144)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 327)] * weight_shared[((((int)threadIdx.x) & 15) + 2160)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 391)] * weight_shared[((((int)threadIdx.x) & 15) + 2160)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 328)] * weight_shared[((((int)threadIdx.x) & 15) + 2176)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 392)] * weight_shared[((((int)threadIdx.x) & 15) + 2176)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 329)] * weight_shared[((((int)threadIdx.x) & 15) + 2192)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 393)] * weight_shared[((((int)threadIdx.x) & 15) + 2192)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 330)] * weight_shared[((((int)threadIdx.x) & 15) + 2208)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 394)] * weight_shared[((((int)threadIdx.x) & 15) + 2208)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 331)] * weight_shared[((((int)threadIdx.x) & 15) + 2224)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 395)] * weight_shared[((((int)threadIdx.x) & 15) + 2224)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 332)] * weight_shared[((((int)threadIdx.x) & 15) + 2240)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 396)] * weight_shared[((((int)threadIdx.x) & 15) + 2240)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 333)] * weight_shared[((((int)threadIdx.x) & 15) + 2256)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 397)] * weight_shared[((((int)threadIdx.x) & 15) + 2256)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 334)] * weight_shared[((((int)threadIdx.x) & 15) + 2272)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 398)] * weight_shared[((((int)threadIdx.x) & 15) + 2272)]));
  conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 335)] * weight_shared[((((int)threadIdx.x) & 15) + 2288)]));
  conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 399)] * weight_shared[((((int)threadIdx.x) & 15) + 2288)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 64)] * weight_shared[((((int)threadIdx.x) & 15) + 512)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 128)] * weight_shared[((((int)threadIdx.x) & 15) + 512)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 65)] * weight_shared[((((int)threadIdx.x) & 15) + 528)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 129)] * weight_shared[((((int)threadIdx.x) & 15) + 528)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 66)] * weight_shared[((((int)threadIdx.x) & 15) + 544)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 130)] * weight_shared[((((int)threadIdx.x) & 15) + 544)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 67)] * weight_shared[((((int)threadIdx.x) & 15) + 560)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 131)] * weight_shared[((((int)threadIdx.x) & 15) + 560)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 68)] * weight_shared[((((int)threadIdx.x) & 15) + 576)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 132)] * weight_shared[((((int)threadIdx.x) & 15) + 576)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 69)] * weight_shared[((((int)threadIdx.x) & 15) + 592)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 133)] * weight_shared[((((int)threadIdx.x) & 15) + 592)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 70)] * weight_shared[((((int)threadIdx.x) & 15) + 608)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 134)] * weight_shared[((((int)threadIdx.x) & 15) + 608)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 71)] * weight_shared[((((int)threadIdx.x) & 15) + 624)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 135)] * weight_shared[((((int)threadIdx.x) & 15) + 624)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 72)] * weight_shared[((((int)threadIdx.x) & 15) + 640)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 136)] * weight_shared[((((int)threadIdx.x) & 15) + 640)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 73)] * weight_shared[((((int)threadIdx.x) & 15) + 656)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 137)] * weight_shared[((((int)threadIdx.x) & 15) + 656)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 74)] * weight_shared[((((int)threadIdx.x) & 15) + 672)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 138)] * weight_shared[((((int)threadIdx.x) & 15) + 672)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 75)] * weight_shared[((((int)threadIdx.x) & 15) + 688)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 139)] * weight_shared[((((int)threadIdx.x) & 15) + 688)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 76)] * weight_shared[((((int)threadIdx.x) & 15) + 704)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 140)] * weight_shared[((((int)threadIdx.x) & 15) + 704)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 77)] * weight_shared[((((int)threadIdx.x) & 15) + 720)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 141)] * weight_shared[((((int)threadIdx.x) & 15) + 720)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 78)] * weight_shared[((((int)threadIdx.x) & 15) + 736)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 142)] * weight_shared[((((int)threadIdx.x) & 15) + 736)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 79)] * weight_shared[((((int)threadIdx.x) & 15) + 752)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 143)] * weight_shared[((((int)threadIdx.x) & 15) + 752)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 208)] * weight_shared[((((int)threadIdx.x) & 15) + 1280)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 272)] * weight_shared[((((int)threadIdx.x) & 15) + 1280)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 209)] * weight_shared[((((int)threadIdx.x) & 15) + 1296)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 273)] * weight_shared[((((int)threadIdx.x) & 15) + 1296)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 210)] * weight_shared[((((int)threadIdx.x) & 15) + 1312)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 274)] * weight_shared[((((int)threadIdx.x) & 15) + 1312)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 211)] * weight_shared[((((int)threadIdx.x) & 15) + 1328)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 275)] * weight_shared[((((int)threadIdx.x) & 15) + 1328)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 212)] * weight_shared[((((int)threadIdx.x) & 15) + 1344)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 276)] * weight_shared[((((int)threadIdx.x) & 15) + 1344)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 213)] * weight_shared[((((int)threadIdx.x) & 15) + 1360)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 277)] * weight_shared[((((int)threadIdx.x) & 15) + 1360)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 214)] * weight_shared[((((int)threadIdx.x) & 15) + 1376)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 278)] * weight_shared[((((int)threadIdx.x) & 15) + 1376)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 215)] * weight_shared[((((int)threadIdx.x) & 15) + 1392)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 279)] * weight_shared[((((int)threadIdx.x) & 15) + 1392)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 216)] * weight_shared[((((int)threadIdx.x) & 15) + 1408)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 280)] * weight_shared[((((int)threadIdx.x) & 15) + 1408)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 217)] * weight_shared[((((int)threadIdx.x) & 15) + 1424)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 281)] * weight_shared[((((int)threadIdx.x) & 15) + 1424)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 218)] * weight_shared[((((int)threadIdx.x) & 15) + 1440)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 282)] * weight_shared[((((int)threadIdx.x) & 15) + 1440)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 219)] * weight_shared[((((int)threadIdx.x) & 15) + 1456)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 283)] * weight_shared[((((int)threadIdx.x) & 15) + 1456)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 220)] * weight_shared[((((int)threadIdx.x) & 15) + 1472)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 284)] * weight_shared[((((int)threadIdx.x) & 15) + 1472)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 221)] * weight_shared[((((int)threadIdx.x) & 15) + 1488)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 285)] * weight_shared[((((int)threadIdx.x) & 15) + 1488)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 222)] * weight_shared[((((int)threadIdx.x) & 15) + 1504)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 286)] * weight_shared[((((int)threadIdx.x) & 15) + 1504)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 223)] * weight_shared[((((int)threadIdx.x) & 15) + 1520)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 287)] * weight_shared[((((int)threadIdx.x) & 15) + 1520)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 352)] * weight_shared[((((int)threadIdx.x) & 15) + 2048)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 416)] * weight_shared[((((int)threadIdx.x) & 15) + 2048)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 353)] * weight_shared[((((int)threadIdx.x) & 15) + 2064)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 417)] * weight_shared[((((int)threadIdx.x) & 15) + 2064)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 354)] * weight_shared[((((int)threadIdx.x) & 15) + 2080)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 418)] * weight_shared[((((int)threadIdx.x) & 15) + 2080)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 355)] * weight_shared[((((int)threadIdx.x) & 15) + 2096)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 419)] * weight_shared[((((int)threadIdx.x) & 15) + 2096)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 356)] * weight_shared[((((int)threadIdx.x) & 15) + 2112)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 420)] * weight_shared[((((int)threadIdx.x) & 15) + 2112)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 357)] * weight_shared[((((int)threadIdx.x) & 15) + 2128)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 421)] * weight_shared[((((int)threadIdx.x) & 15) + 2128)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 358)] * weight_shared[((((int)threadIdx.x) & 15) + 2144)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 422)] * weight_shared[((((int)threadIdx.x) & 15) + 2144)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 359)] * weight_shared[((((int)threadIdx.x) & 15) + 2160)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 423)] * weight_shared[((((int)threadIdx.x) & 15) + 2160)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 360)] * weight_shared[((((int)threadIdx.x) & 15) + 2176)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 424)] * weight_shared[((((int)threadIdx.x) & 15) + 2176)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 361)] * weight_shared[((((int)threadIdx.x) & 15) + 2192)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 425)] * weight_shared[((((int)threadIdx.x) & 15) + 2192)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 362)] * weight_shared[((((int)threadIdx.x) & 15) + 2208)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 426)] * weight_shared[((((int)threadIdx.x) & 15) + 2208)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 363)] * weight_shared[((((int)threadIdx.x) & 15) + 2224)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 427)] * weight_shared[((((int)threadIdx.x) & 15) + 2224)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 364)] * weight_shared[((((int)threadIdx.x) & 15) + 2240)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 428)] * weight_shared[((((int)threadIdx.x) & 15) + 2240)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 365)] * weight_shared[((((int)threadIdx.x) & 15) + 2256)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 429)] * weight_shared[((((int)threadIdx.x) & 15) + 2256)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 366)] * weight_shared[((((int)threadIdx.x) & 15) + 2272)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 430)] * weight_shared[((((int)threadIdx.x) & 15) + 2272)]));
  conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 367)] * weight_shared[((((int)threadIdx.x) & 15) + 2288)]));
  conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[(((((int)threadIdx.x) >> 4) * 288) + 431)] * weight_shared[((((int)threadIdx.x) & 15) + 2288)]));
  conv2d_nhwc[((((((((int)blockIdx.x) / 56) * 14336) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((((int)blockIdx.x) / 56) * 14336) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 256)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((((int)blockIdx.x) / 56) * 14336) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 128)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((((int)blockIdx.x) / 56) * 14336) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 56) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 384)] = conv2d_nhwc_local[3];
}

