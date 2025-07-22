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
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ depth_conv2d_nhwc, half* __restrict__ placeholder, half* __restrict__ placeholder_1);
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ depth_conv2d_nhwc, half* __restrict__ placeholder, half* __restrict__ placeholder_1) {
  half depth_conv2d_nhwc_local[16];
  __shared__ half PadInput_shared[3264];
  __shared__ half placeholder_shared[576];
  depth_conv2d_nhwc_local[0] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[1] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[2] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[3] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[4] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[5] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[6] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[7] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[8] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[9] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[10] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[11] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[12] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[13] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[14] = __float2half_rn(0.000000e+00f);
  depth_conv2d_nhwc_local[15] = __float2half_rn(0.000000e+00f);
  half condval;
  if (((56 <= ((int)blockIdx.x)) && (1 <= (((int)blockIdx.x) % 56)))) {
    condval = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 7232)];
  } else {
    condval = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[((int)threadIdx.x)] = condval;
  half condval_1;
  if (((56 <= ((int)blockIdx.x)) && (1 <= (((int)blockIdx.x) % 56)))) {
    condval_1 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 7200)];
  } else {
    condval_1 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 32)] = condval_1;
  half condval_2;
  if ((56 <= ((int)blockIdx.x))) {
    condval_2 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 7168)];
  } else {
    condval_2 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 64)] = condval_2;
  half condval_3;
  if ((56 <= ((int)blockIdx.x))) {
    condval_3 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 7136)];
  } else {
    condval_3 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 96)] = condval_3;
  half condval_4;
  if ((56 <= ((int)blockIdx.x))) {
    condval_4 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 7104)];
  } else {
    condval_4 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 128)] = condval_4;
  half condval_5;
  if ((56 <= ((int)blockIdx.x))) {
    condval_5 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 7072)];
  } else {
    condval_5 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 160)] = condval_5;
  half condval_6;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_6 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 64)];
  } else {
    condval_6 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 192)] = condval_6;
  half condval_7;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_7 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) - 32)];
  } else {
    condval_7 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 224)] = condval_7;
  PadInput_shared[(((int)threadIdx.x) + 256)] = placeholder[((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x))];
  PadInput_shared[(((int)threadIdx.x) + 288)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 32)];
  PadInput_shared[(((int)threadIdx.x) + 320)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 64)];
  PadInput_shared[(((int)threadIdx.x) + 352)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 96)];
  half condval_8;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_8 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 7104)];
  } else {
    condval_8 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 384)] = condval_8;
  half condval_9;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_9 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 7136)];
  } else {
    condval_9 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 416)] = condval_9;
  PadInput_shared[(((int)threadIdx.x) + 448)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 7168)];
  PadInput_shared[(((int)threadIdx.x) + 480)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 7200)];
  PadInput_shared[(((int)threadIdx.x) + 512)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 7232)];
  PadInput_shared[(((int)threadIdx.x) + 544)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 7264)];
  half condval_10;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_10 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 14272)];
  } else {
    condval_10 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 576)] = condval_10;
  half condval_11;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_11 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 14304)];
  } else {
    condval_11 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 608)] = condval_11;
  PadInput_shared[(((int)threadIdx.x) + 640)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 14336)];
  PadInput_shared[(((int)threadIdx.x) + 672)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 14368)];
  PadInput_shared[(((int)threadIdx.x) + 704)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 14400)];
  PadInput_shared[(((int)threadIdx.x) + 736)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 14432)];
  half condval_12;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_12 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 21440)];
  } else {
    condval_12 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 768)] = condval_12;
  half condval_13;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_13 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 21472)];
  } else {
    condval_13 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 800)] = condval_13;
  PadInput_shared[(((int)threadIdx.x) + 832)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 21504)];
  PadInput_shared[(((int)threadIdx.x) + 864)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 21536)];
  PadInput_shared[(((int)threadIdx.x) + 896)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 21568)];
  PadInput_shared[(((int)threadIdx.x) + 928)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 21600)];
  half condval_14;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_14 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 28608)];
  } else {
    condval_14 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 960)] = condval_14;
  half condval_15;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_15 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 28640)];
  } else {
    condval_15 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 992)] = condval_15;
  PadInput_shared[(((int)threadIdx.x) + 1024)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 28672)];
  PadInput_shared[(((int)threadIdx.x) + 1056)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 28704)];
  PadInput_shared[(((int)threadIdx.x) + 1088)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 28736)];
  PadInput_shared[(((int)threadIdx.x) + 1120)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 28768)];
  half condval_16;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_16 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 35776)];
  } else {
    condval_16 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1152)] = condval_16;
  half condval_17;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_17 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 35808)];
  } else {
    condval_17 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1184)] = condval_17;
  PadInput_shared[(((int)threadIdx.x) + 1216)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 35840)];
  PadInput_shared[(((int)threadIdx.x) + 1248)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 35872)];
  PadInput_shared[(((int)threadIdx.x) + 1280)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 35904)];
  PadInput_shared[(((int)threadIdx.x) + 1312)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 35936)];
  half condval_18;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_18 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 42944)];
  } else {
    condval_18 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1344)] = condval_18;
  half condval_19;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_19 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 42976)];
  } else {
    condval_19 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1376)] = condval_19;
  PadInput_shared[(((int)threadIdx.x) + 1408)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 43008)];
  PadInput_shared[(((int)threadIdx.x) + 1440)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 43040)];
  PadInput_shared[(((int)threadIdx.x) + 1472)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 43072)];
  PadInput_shared[(((int)threadIdx.x) + 1504)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 43104)];
  half condval_20;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_20 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 50112)];
  } else {
    condval_20 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1536)] = condval_20;
  half condval_21;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_21 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 50144)];
  } else {
    condval_21 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1568)] = condval_21;
  PadInput_shared[(((int)threadIdx.x) + 1600)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 50176)];
  PadInput_shared[(((int)threadIdx.x) + 1632)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 50208)];
  PadInput_shared[(((int)threadIdx.x) + 1664)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 50240)];
  PadInput_shared[(((int)threadIdx.x) + 1696)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 50272)];
  half condval_22;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_22 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 57280)];
  } else {
    condval_22 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1728)] = condval_22;
  half condval_23;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_23 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 57312)];
  } else {
    condval_23 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1760)] = condval_23;
  PadInput_shared[(((int)threadIdx.x) + 1792)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 57344)];
  PadInput_shared[(((int)threadIdx.x) + 1824)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 57376)];
  PadInput_shared[(((int)threadIdx.x) + 1856)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 57408)];
  PadInput_shared[(((int)threadIdx.x) + 1888)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 57440)];
  half condval_24;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_24 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 64448)];
  } else {
    condval_24 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1920)] = condval_24;
  half condval_25;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_25 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 64480)];
  } else {
    condval_25 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 1952)] = condval_25;
  PadInput_shared[(((int)threadIdx.x) + 1984)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 64512)];
  PadInput_shared[(((int)threadIdx.x) + 2016)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 64544)];
  PadInput_shared[(((int)threadIdx.x) + 2048)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 64576)];
  PadInput_shared[(((int)threadIdx.x) + 2080)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 64608)];
  half condval_26;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_26 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 71616)];
  } else {
    condval_26 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2112)] = condval_26;
  half condval_27;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_27 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 71648)];
  } else {
    condval_27 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2144)] = condval_27;
  PadInput_shared[(((int)threadIdx.x) + 2176)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 71680)];
  PadInput_shared[(((int)threadIdx.x) + 2208)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 71712)];
  PadInput_shared[(((int)threadIdx.x) + 2240)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 71744)];
  PadInput_shared[(((int)threadIdx.x) + 2272)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 71776)];
  half condval_28;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_28 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 78784)];
  } else {
    condval_28 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2304)] = condval_28;
  half condval_29;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_29 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 78816)];
  } else {
    condval_29 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2336)] = condval_29;
  PadInput_shared[(((int)threadIdx.x) + 2368)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 78848)];
  PadInput_shared[(((int)threadIdx.x) + 2400)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 78880)];
  PadInput_shared[(((int)threadIdx.x) + 2432)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 78912)];
  PadInput_shared[(((int)threadIdx.x) + 2464)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 78944)];
  half condval_30;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_30 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 85952)];
  } else {
    condval_30 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2496)] = condval_30;
  half condval_31;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_31 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 85984)];
  } else {
    condval_31 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2528)] = condval_31;
  PadInput_shared[(((int)threadIdx.x) + 2560)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 86016)];
  PadInput_shared[(((int)threadIdx.x) + 2592)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 86048)];
  PadInput_shared[(((int)threadIdx.x) + 2624)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 86080)];
  PadInput_shared[(((int)threadIdx.x) + 2656)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 86112)];
  half condval_32;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_32 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 93120)];
  } else {
    condval_32 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2688)] = condval_32;
  half condval_33;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_33 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 93152)];
  } else {
    condval_33 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2720)] = condval_33;
  PadInput_shared[(((int)threadIdx.x) + 2752)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 93184)];
  PadInput_shared[(((int)threadIdx.x) + 2784)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 93216)];
  PadInput_shared[(((int)threadIdx.x) + 2816)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 93248)];
  PadInput_shared[(((int)threadIdx.x) + 2848)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 93280)];
  half condval_34;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_34 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 100288)];
  } else {
    condval_34 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2880)] = condval_34;
  half condval_35;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_35 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 100320)];
  } else {
    condval_35 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 2912)] = condval_35;
  PadInput_shared[(((int)threadIdx.x) + 2944)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 100352)];
  PadInput_shared[(((int)threadIdx.x) + 2976)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 100384)];
  PadInput_shared[(((int)threadIdx.x) + 3008)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 100416)];
  PadInput_shared[(((int)threadIdx.x) + 3040)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 100448)];
  half condval_36;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_36 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 107456)];
  } else {
    condval_36 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 3072)] = condval_36;
  half condval_37;
  if ((1 <= (((int)blockIdx.x) % 56))) {
    condval_37 = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 107488)];
  } else {
    condval_37 = __float2half_rn(0.000000e+00f);
  }
  PadInput_shared[(((int)threadIdx.x) + 3104)] = condval_37;
  PadInput_shared[(((int)threadIdx.x) + 3136)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 107520)];
  PadInput_shared[(((int)threadIdx.x) + 3168)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 107552)];
  PadInput_shared[(((int)threadIdx.x) + 3200)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 107584)];
  PadInput_shared[(((int)threadIdx.x) + 3232)] = placeholder[(((((((int)blockIdx.x) / 56) * 114688) + ((((int)blockIdx.x) % 56) * 128)) + ((int)threadIdx.x)) + 107616)];
  *(uint4*)(placeholder_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(placeholder_1 + (((int)threadIdx.x) * 8));
  *(uint4*)(placeholder_shared + ((((int)threadIdx.x) * 8) + 256)) = *(uint4*)(placeholder_1 + ((((int)threadIdx.x) * 8) + 256));
  if (((int)threadIdx.x) < 8) {
    *(uint4*)(placeholder_shared + ((((int)threadIdx.x) * 8) + 512)) = *(uint4*)(placeholder_1 + ((((int)threadIdx.x) * 8) + 512));
  }
  __syncthreads();
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[(((int)threadIdx.x) * 2)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 384)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 385)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 768)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 769)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1152)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1153)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1536)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1537)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1920)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1921)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2304)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2305)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2688)] * placeholder_shared[(((int)threadIdx.x) * 2)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2689)] * placeholder_shared[((((int)threadIdx.x) * 2) + 1)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 64)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 65)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 448)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 449)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 832)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 833)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1216)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1217)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1600)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1601)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1984)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1985)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2368)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2369)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2752)] * placeholder_shared[((((int)threadIdx.x) * 2) + 64)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2753)] * placeholder_shared[((((int)threadIdx.x) * 2) + 65)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 128)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 129)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 512)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 513)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 896)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 897)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1280)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1281)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1664)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1665)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2048)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2049)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2432)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2433)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2816)] * placeholder_shared[((((int)threadIdx.x) * 2) + 128)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2817)] * placeholder_shared[((((int)threadIdx.x) * 2) + 129)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 192)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 193)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 576)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 577)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 960)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 961)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1344)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1345)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1728)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1729)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2112)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2113)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2496)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2497)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2880)] * placeholder_shared[((((int)threadIdx.x) * 2) + 192)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2881)] * placeholder_shared[((((int)threadIdx.x) * 2) + 193)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 256)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 257)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 640)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 641)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1024)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1025)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1408)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1409)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1792)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1793)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2176)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2177)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2560)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2561)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2944)] * placeholder_shared[((((int)threadIdx.x) * 2) + 256)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2945)] * placeholder_shared[((((int)threadIdx.x) * 2) + 257)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 320)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 321)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 704)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 705)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1088)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1089)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1472)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1473)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1856)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1857)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2240)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2241)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2624)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2625)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3008)] * placeholder_shared[((((int)threadIdx.x) * 2) + 320)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3009)] * placeholder_shared[((((int)threadIdx.x) * 2) + 321)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 384)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 385)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 768)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 769)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1152)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1153)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1536)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1537)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1920)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1921)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2304)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2305)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2688)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2689)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3072)] * placeholder_shared[((((int)threadIdx.x) * 2) + 384)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3073)] * placeholder_shared[((((int)threadIdx.x) * 2) + 385)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 448)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 449)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 832)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 833)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1216)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1217)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1600)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1601)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1984)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1985)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2368)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2369)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2752)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2753)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3136)] * placeholder_shared[((((int)threadIdx.x) * 2) + 448)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3137)] * placeholder_shared[((((int)threadIdx.x) * 2) + 449)]));
  depth_conv2d_nhwc_local[0] = (depth_conv2d_nhwc_local[0] + (PadInput_shared[((((int)threadIdx.x) * 2) + 512)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[1] = (depth_conv2d_nhwc_local[1] + (PadInput_shared[((((int)threadIdx.x) * 2) + 513)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc_local[2] = (depth_conv2d_nhwc_local[2] + (PadInput_shared[((((int)threadIdx.x) * 2) + 896)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[3] = (depth_conv2d_nhwc_local[3] + (PadInput_shared[((((int)threadIdx.x) * 2) + 897)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc_local[4] = (depth_conv2d_nhwc_local[4] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1280)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[5] = (depth_conv2d_nhwc_local[5] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1281)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc_local[6] = (depth_conv2d_nhwc_local[6] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1664)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[7] = (depth_conv2d_nhwc_local[7] + (PadInput_shared[((((int)threadIdx.x) * 2) + 1665)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc_local[8] = (depth_conv2d_nhwc_local[8] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2048)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[9] = (depth_conv2d_nhwc_local[9] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2049)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc_local[10] = (depth_conv2d_nhwc_local[10] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2432)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[11] = (depth_conv2d_nhwc_local[11] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2433)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc_local[12] = (depth_conv2d_nhwc_local[12] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2816)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[13] = (depth_conv2d_nhwc_local[13] + (PadInput_shared[((((int)threadIdx.x) * 2) + 2817)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc_local[14] = (depth_conv2d_nhwc_local[14] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3200)] * placeholder_shared[((((int)threadIdx.x) * 2) + 512)]));
  depth_conv2d_nhwc_local[15] = (depth_conv2d_nhwc_local[15] + (PadInput_shared[((((int)threadIdx.x) * 2) + 3201)] * placeholder_shared[((((int)threadIdx.x) * 2) + 513)]));
  depth_conv2d_nhwc[((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2))] = depth_conv2d_nhwc_local[0];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 1)] = depth_conv2d_nhwc_local[1];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 3584)] = depth_conv2d_nhwc_local[2];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 3585)] = depth_conv2d_nhwc_local[3];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 7168)] = depth_conv2d_nhwc_local[4];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 7169)] = depth_conv2d_nhwc_local[5];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 10752)] = depth_conv2d_nhwc_local[6];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 10753)] = depth_conv2d_nhwc_local[7];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 14336)] = depth_conv2d_nhwc_local[8];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 14337)] = depth_conv2d_nhwc_local[9];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 17920)] = depth_conv2d_nhwc_local[10];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 17921)] = depth_conv2d_nhwc_local[11];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 21504)] = depth_conv2d_nhwc_local[12];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 21505)] = depth_conv2d_nhwc_local[13];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 25088)] = depth_conv2d_nhwc_local[14];
  depth_conv2d_nhwc[(((((((int)blockIdx.x) / 56) * 28672) + ((((int)blockIdx.x) % 56) * 64)) + (((int)threadIdx.x) * 2)) + 25089)] = depth_conv2d_nhwc_local[15];
}

