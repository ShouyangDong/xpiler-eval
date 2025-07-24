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
extern "C" __global__ void __launch_bounds__(48) main_kernel(half* __restrict__ conv2d_transpose_nhwc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(48) main_kernel(half* __restrict__ conv2d_transpose_nhwc, half* __restrict__ inputs, half* __restrict__ weight) {
  half conv2d_transpose_nhwc_local[4];
  __shared__ half PadInput_shared[4608];
  __shared__ half weight_shared[6144];
  conv2d_transpose_nhwc_local[0] = __float2half_rn(0.000000e+00f);
  conv2d_transpose_nhwc_local[1] = __float2half_rn(0.000000e+00f);
  conv2d_transpose_nhwc_local[2] = __float2half_rn(0.000000e+00f);
  conv2d_transpose_nhwc_local[3] = __float2half_rn(0.000000e+00f);
  uint4 condval;
  if ((((1 <= (((((int)blockIdx.x) >> 3) * 4) + (((int)threadIdx.x) / 24))) && (1 <= (((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)))) && ((((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)) < 33))) {
    condval = *(uint4*)(inputs + (((((((((int)blockIdx.x) >> 3) * 8192) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 7) * 256)) + (((((int)threadIdx.x) % 24) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 8)) - 2112));
  } else {
    condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(PadInput_shared + (((int)threadIdx.x) * 8)) = condval;
  uint4 condval_1;
  if (((1 <= (((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2))) && ((((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)) < 33))) {
    condval_1 = *(uint4*)(inputs + (((((((((int)blockIdx.x) >> 3) * 8192) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 7) * 256)) + (((((int)threadIdx.x) % 24) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 8)) + 1984));
  } else {
    condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(PadInput_shared + ((((int)threadIdx.x) * 8) + 384)) = condval_1;
  uint4 condval_2;
  if (((((((((int)blockIdx.x) >> 3) * 4) + (((int)threadIdx.x) / 24)) < 29) && (1 <= (((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)))) && ((((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)) < 33))) {
    condval_2 = *(uint4*)(inputs + (((((((((int)blockIdx.x) >> 3) * 8192) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 7) * 256)) + (((((int)threadIdx.x) % 24) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 8)) + 6080));
  } else {
    condval_2 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(PadInput_shared + ((((int)threadIdx.x) * 8) + 768)) = condval_2;
  weight_shared[((int)threadIdx.x)] = weight[((int)threadIdx.x)];
  weight_shared[(((int)threadIdx.x) + 48)] = weight[(((int)threadIdx.x) + 48)];
  weight_shared[(((int)threadIdx.x) + 96)] = weight[(((int)threadIdx.x) + 192)];
  weight_shared[(((int)threadIdx.x) + 144)] = weight[(((int)threadIdx.x) + 240)];
  weight_shared[(((int)threadIdx.x) + 192)] = weight[(((int)threadIdx.x) + 384)];
  weight_shared[(((int)threadIdx.x) + 240)] = weight[(((int)threadIdx.x) + 432)];
  weight_shared[(((int)threadIdx.x) + 288)] = weight[(((int)threadIdx.x) + 576)];
  weight_shared[(((int)threadIdx.x) + 336)] = weight[(((int)threadIdx.x) + 624)];
  weight_shared[(((int)threadIdx.x) + 384)] = weight[(((int)threadIdx.x) + 768)];
  weight_shared[(((int)threadIdx.x) + 432)] = weight[(((int)threadIdx.x) + 816)];
  weight_shared[(((int)threadIdx.x) + 480)] = weight[(((int)threadIdx.x) + 960)];
  weight_shared[(((int)threadIdx.x) + 528)] = weight[(((int)threadIdx.x) + 1008)];
  weight_shared[(((int)threadIdx.x) + 576)] = weight[(((int)threadIdx.x) + 1152)];
  weight_shared[(((int)threadIdx.x) + 624)] = weight[(((int)threadIdx.x) + 1200)];
  weight_shared[(((int)threadIdx.x) + 672)] = weight[(((int)threadIdx.x) + 1344)];
  weight_shared[(((int)threadIdx.x) + 720)] = weight[(((int)threadIdx.x) + 1392)];
  weight_shared[(((int)threadIdx.x) + 768)] = weight[(((int)threadIdx.x) + 1536)];
  weight_shared[(((int)threadIdx.x) + 816)] = weight[(((int)threadIdx.x) + 1584)];
  weight_shared[(((int)threadIdx.x) + 864)] = weight[(((int)threadIdx.x) + 1728)];
  weight_shared[(((int)threadIdx.x) + 912)] = weight[(((int)threadIdx.x) + 1776)];
  weight_shared[(((int)threadIdx.x) + 960)] = weight[(((int)threadIdx.x) + 1920)];
  weight_shared[(((int)threadIdx.x) + 1008)] = weight[(((int)threadIdx.x) + 1968)];
  weight_shared[(((int)threadIdx.x) + 1056)] = weight[(((int)threadIdx.x) + 2112)];
  weight_shared[(((int)threadIdx.x) + 1104)] = weight[(((int)threadIdx.x) + 2160)];
  weight_shared[(((int)threadIdx.x) + 1152)] = weight[(((int)threadIdx.x) + 2304)];
  weight_shared[(((int)threadIdx.x) + 1200)] = weight[(((int)threadIdx.x) + 2352)];
  weight_shared[(((int)threadIdx.x) + 1248)] = weight[(((int)threadIdx.x) + 2496)];
  weight_shared[(((int)threadIdx.x) + 1296)] = weight[(((int)threadIdx.x) + 2544)];
  weight_shared[(((int)threadIdx.x) + 1344)] = weight[(((int)threadIdx.x) + 2688)];
  weight_shared[(((int)threadIdx.x) + 1392)] = weight[(((int)threadIdx.x) + 2736)];
  weight_shared[(((int)threadIdx.x) + 1440)] = weight[(((int)threadIdx.x) + 2880)];
  weight_shared[(((int)threadIdx.x) + 1488)] = weight[(((int)threadIdx.x) + 2928)];
__asm__ __volatile__("cp.async.commit_group;");

  uint4 condval_3;
  if ((((1 <= (((((int)blockIdx.x) >> 3) * 4) + (((int)threadIdx.x) / 24))) && (1 <= (((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)))) && ((((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)) < 33))) {
    condval_3 = *(uint4*)(inputs + (((((((((int)blockIdx.x) >> 3) * 8192) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 7) * 256)) + (((((int)threadIdx.x) % 24) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 8)) - 2080));
  } else {
    condval_3 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(PadInput_shared + ((((int)threadIdx.x) * 8) + 1152)) = condval_3;
  uint4 condval_4;
  if (((1 <= (((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2))) && ((((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)) < 33))) {
    condval_4 = *(uint4*)(inputs + (((((((((int)blockIdx.x) >> 3) * 8192) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 7) * 256)) + (((((int)threadIdx.x) % 24) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 8)) + 2016));
  } else {
    condval_4 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(PadInput_shared + ((((int)threadIdx.x) * 8) + 1536)) = condval_4;
  uint4 condval_5;
  if (((((((((int)blockIdx.x) >> 3) * 4) + (((int)threadIdx.x) / 24)) < 29) && (1 <= (((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)))) && ((((((int)blockIdx.x) & 7) * 4) + ((((int)threadIdx.x) % 24) >> 2)) < 33))) {
    condval_5 = *(uint4*)(inputs + (((((((((int)blockIdx.x) >> 3) * 8192) + ((((int)threadIdx.x) / 24) * 2048)) + ((((int)blockIdx.x) & 7) * 256)) + (((((int)threadIdx.x) % 24) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 8)) + 6112));
  } else {
    condval_5 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(PadInput_shared + ((((int)threadIdx.x) * 8) + 1920)) = condval_5;
  weight_shared[(((int)threadIdx.x) + 1536)] = weight[(((int)threadIdx.x) + 96)];
  weight_shared[(((int)threadIdx.x) + 1584)] = weight[(((int)threadIdx.x) + 144)];
  weight_shared[(((int)threadIdx.x) + 1632)] = weight[(((int)threadIdx.x) + 288)];
  weight_shared[(((int)threadIdx.x) + 1680)] = weight[(((int)threadIdx.x) + 336)];
  weight_shared[(((int)threadIdx.x) + 1728)] = weight[(((int)threadIdx.x) + 480)];
  weight_shared[(((int)threadIdx.x) + 1776)] = weight[(((int)threadIdx.x) + 528)];
  weight_shared[(((int)threadIdx.x) + 1824)] = weight[(((int)threadIdx.x) + 672)];
  weight_shared[(((int)threadIdx.x) + 1872)] = weight[(((int)threadIdx.x) + 720)];
  weight_shared[(((int)threadIdx.x) + 1920)] = weight[(((int)threadIdx.x) + 864)];
  weight_shared[(((int)threadIdx.x) + 1968)] = weight[(((int)threadIdx.x) + 912)];
  weight_shared[(((int)threadIdx.x) + 2016)] = weight[(((int)threadIdx.x) + 1056)];
  weight_shared[(((int)threadIdx.x) + 2064)] = weight[(((int)threadIdx.x) + 1104)];
  weight_shared[(((int)threadIdx.x) + 2112)] = weight[(((int)threadIdx.x) + 1248)];
  weight_shared[(((int)threadIdx.x) + 2160)] = weight[(((int)threadIdx.x) + 1296)];
  weight_shared[(((int)threadIdx.x) + 2208)] = weight[(((int)threadIdx.x) + 1440)];
  weight_shared[(((int)threadIdx.x) + 2256)] = weight[(((int)threadIdx.x) + 1488)];
  weight_shared[(((int)threadIdx.x) + 2304)] = weight[(((int)threadIdx.x) + 1632)];
  weight_shared[(((int)threadIdx.x) + 2352)] = weight[(((int)threadIdx.x) + 1680)];
  weight_shared[(((int)threadIdx.x) + 2400)] = weight[(((int)threadIdx.x) + 1824)];
  weight_shared[(((int)threadIdx.x) + 2448)] = weight[(((int)threadIdx.x) + 1872)];
  weight_shared[(((int)threadIdx.x) + 2496)] = weight[(((int)threadIdx.x) + 2016)];
  weight_shared[(((int)threadIdx.x) + 2544)] = weight[(((int)threadIdx.x) + 2064)];
  weight_shared[(((int)threadIdx.x) + 2592)] = weight[(((int)threadIdx.x) + 2208)];
  weight_shared[(((int)threadIdx.x) + 2640)] = weight[(((int)threadIdx.x) + 2256)];
  weight_shared[(((int)threadIdx.x) + 2688)] = weight[(((int)threadIdx.x) + 2400)];
  weight_shared[(((int)threadIdx.x) + 2736)] = weight[(((int)threadIdx.x) + 2448)];
  weight_shared[(((int)threadIdx.x) + 2784)] = weight[(((int)threadIdx.x) + 2592)];
  weight_shared[(((int)threadIdx.x) + 2832)] = weight[(((int)threadIdx.x) + 2640)];
  weight_shared[(((int)threadIdx.x) + 2880)] = weight[(((int)threadIdx.x) + 2784)];
  weight_shared[(((int)threadIdx.x) + 2928)] = weight[(((int)threadIdx.x) + 2832)];
  weight_shared[(((int)threadIdx.x) + 2976)] = weight[(((int)threadIdx.x) + 2976)];
  weight_shared[(((int)threadIdx.x) + 3024)] = weight[(((int)threadIdx.x) + 3024)];
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int rw_1 = 0; rw_1 < 2; ++rw_1) {
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32))] * weight_shared[(((((int)threadIdx.x) % 3) + 1440) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 1)] * weight_shared[(((((int)threadIdx.x) % 3) + 1443) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 32)] * weight_shared[(((((int)threadIdx.x) % 3) + 1344) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 33)] * weight_shared[(((((int)threadIdx.x) % 3) + 1347) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 192)] * weight_shared[(((((int)threadIdx.x) % 3) + 672) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 193)] * weight_shared[(((((int)threadIdx.x) % 3) + 675) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 224)] * weight_shared[(((((int)threadIdx.x) % 3) + 576) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 225)] * weight_shared[(((((int)threadIdx.x) % 3) + 579) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 192)] * weight_shared[(((((int)threadIdx.x) % 3) + 1056) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 193)] * weight_shared[(((((int)threadIdx.x) % 3) + 1059) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 224)] * weight_shared[(((((int)threadIdx.x) % 3) + 960) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 225)] * weight_shared[(((((int)threadIdx.x) % 3) + 963) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 384)] * weight_shared[(((((int)threadIdx.x) % 3) + 288) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 385)] * weight_shared[(((((int)threadIdx.x) % 3) + 291) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 416)] * weight_shared[(((((int)threadIdx.x) % 3) + 192) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 417)] * weight_shared[(((((int)threadIdx.x) % 3) + 195) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 2)] * weight_shared[(((((int)threadIdx.x) % 3) + 1446) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 3)] * weight_shared[(((((int)threadIdx.x) % 3) + 1449) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 34)] * weight_shared[(((((int)threadIdx.x) % 3) + 1350) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 35)] * weight_shared[(((((int)threadIdx.x) % 3) + 1353) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 194)] * weight_shared[(((((int)threadIdx.x) % 3) + 678) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 195)] * weight_shared[(((((int)threadIdx.x) % 3) + 681) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 226)] * weight_shared[(((((int)threadIdx.x) % 3) + 582) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 227)] * weight_shared[(((((int)threadIdx.x) % 3) + 585) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 194)] * weight_shared[(((((int)threadIdx.x) % 3) + 1062) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 195)] * weight_shared[(((((int)threadIdx.x) % 3) + 1065) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 226)] * weight_shared[(((((int)threadIdx.x) % 3) + 966) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 227)] * weight_shared[(((((int)threadIdx.x) % 3) + 969) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 386)] * weight_shared[(((((int)threadIdx.x) % 3) + 294) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 387)] * weight_shared[(((((int)threadIdx.x) % 3) + 297) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 418)] * weight_shared[(((((int)threadIdx.x) % 3) + 198) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 419)] * weight_shared[(((((int)threadIdx.x) % 3) + 201) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 4)] * weight_shared[(((((int)threadIdx.x) % 3) + 1452) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 5)] * weight_shared[(((((int)threadIdx.x) % 3) + 1455) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 36)] * weight_shared[(((((int)threadIdx.x) % 3) + 1356) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 37)] * weight_shared[(((((int)threadIdx.x) % 3) + 1359) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 196)] * weight_shared[(((((int)threadIdx.x) % 3) + 684) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 197)] * weight_shared[(((((int)threadIdx.x) % 3) + 687) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 228)] * weight_shared[(((((int)threadIdx.x) % 3) + 588) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 229)] * weight_shared[(((((int)threadIdx.x) % 3) + 591) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 196)] * weight_shared[(((((int)threadIdx.x) % 3) + 1068) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 197)] * weight_shared[(((((int)threadIdx.x) % 3) + 1071) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 228)] * weight_shared[(((((int)threadIdx.x) % 3) + 972) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 229)] * weight_shared[(((((int)threadIdx.x) % 3) + 975) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 388)] * weight_shared[(((((int)threadIdx.x) % 3) + 300) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 389)] * weight_shared[(((((int)threadIdx.x) % 3) + 303) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 420)] * weight_shared[(((((int)threadIdx.x) % 3) + 204) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 421)] * weight_shared[(((((int)threadIdx.x) % 3) + 207) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 6)] * weight_shared[(((((int)threadIdx.x) % 3) + 1458) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 7)] * weight_shared[(((((int)threadIdx.x) % 3) + 1461) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 38)] * weight_shared[(((((int)threadIdx.x) % 3) + 1362) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 39)] * weight_shared[(((((int)threadIdx.x) % 3) + 1365) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 198)] * weight_shared[(((((int)threadIdx.x) % 3) + 690) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 199)] * weight_shared[(((((int)threadIdx.x) % 3) + 693) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 230)] * weight_shared[(((((int)threadIdx.x) % 3) + 594) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 231)] * weight_shared[(((((int)threadIdx.x) % 3) + 597) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 198)] * weight_shared[(((((int)threadIdx.x) % 3) + 1074) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 199)] * weight_shared[(((((int)threadIdx.x) % 3) + 1077) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 230)] * weight_shared[(((((int)threadIdx.x) % 3) + 978) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 231)] * weight_shared[(((((int)threadIdx.x) % 3) + 981) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 390)] * weight_shared[(((((int)threadIdx.x) % 3) + 306) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 391)] * weight_shared[(((((int)threadIdx.x) % 3) + 309) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 422)] * weight_shared[(((((int)threadIdx.x) % 3) + 210) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 423)] * weight_shared[(((((int)threadIdx.x) % 3) + 213) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 8)] * weight_shared[(((((int)threadIdx.x) % 3) + 1464) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 9)] * weight_shared[(((((int)threadIdx.x) % 3) + 1467) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 40)] * weight_shared[(((((int)threadIdx.x) % 3) + 1368) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 41)] * weight_shared[(((((int)threadIdx.x) % 3) + 1371) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 200)] * weight_shared[(((((int)threadIdx.x) % 3) + 696) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 201)] * weight_shared[(((((int)threadIdx.x) % 3) + 699) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 232)] * weight_shared[(((((int)threadIdx.x) % 3) + 600) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 233)] * weight_shared[(((((int)threadIdx.x) % 3) + 603) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 200)] * weight_shared[(((((int)threadIdx.x) % 3) + 1080) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 201)] * weight_shared[(((((int)threadIdx.x) % 3) + 1083) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 232)] * weight_shared[(((((int)threadIdx.x) % 3) + 984) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 233)] * weight_shared[(((((int)threadIdx.x) % 3) + 987) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 392)] * weight_shared[(((((int)threadIdx.x) % 3) + 312) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 393)] * weight_shared[(((((int)threadIdx.x) % 3) + 315) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 424)] * weight_shared[(((((int)threadIdx.x) % 3) + 216) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 425)] * weight_shared[(((((int)threadIdx.x) % 3) + 219) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 10)] * weight_shared[(((((int)threadIdx.x) % 3) + 1470) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 11)] * weight_shared[(((((int)threadIdx.x) % 3) + 1473) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 42)] * weight_shared[(((((int)threadIdx.x) % 3) + 1374) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 43)] * weight_shared[(((((int)threadIdx.x) % 3) + 1377) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 202)] * weight_shared[(((((int)threadIdx.x) % 3) + 702) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 203)] * weight_shared[(((((int)threadIdx.x) % 3) + 705) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 234)] * weight_shared[(((((int)threadIdx.x) % 3) + 606) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 235)] * weight_shared[(((((int)threadIdx.x) % 3) + 609) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 202)] * weight_shared[(((((int)threadIdx.x) % 3) + 1086) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 203)] * weight_shared[(((((int)threadIdx.x) % 3) + 1089) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 234)] * weight_shared[(((((int)threadIdx.x) % 3) + 990) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 235)] * weight_shared[(((((int)threadIdx.x) % 3) + 993) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 394)] * weight_shared[(((((int)threadIdx.x) % 3) + 318) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 395)] * weight_shared[(((((int)threadIdx.x) % 3) + 321) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 426)] * weight_shared[(((((int)threadIdx.x) % 3) + 222) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 427)] * weight_shared[(((((int)threadIdx.x) % 3) + 225) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 12)] * weight_shared[(((((int)threadIdx.x) % 3) + 1476) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 13)] * weight_shared[(((((int)threadIdx.x) % 3) + 1479) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 44)] * weight_shared[(((((int)threadIdx.x) % 3) + 1380) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 45)] * weight_shared[(((((int)threadIdx.x) % 3) + 1383) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 204)] * weight_shared[(((((int)threadIdx.x) % 3) + 708) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 205)] * weight_shared[(((((int)threadIdx.x) % 3) + 711) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 236)] * weight_shared[(((((int)threadIdx.x) % 3) + 612) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 237)] * weight_shared[(((((int)threadIdx.x) % 3) + 615) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 204)] * weight_shared[(((((int)threadIdx.x) % 3) + 1092) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 205)] * weight_shared[(((((int)threadIdx.x) % 3) + 1095) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 236)] * weight_shared[(((((int)threadIdx.x) % 3) + 996) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 237)] * weight_shared[(((((int)threadIdx.x) % 3) + 999) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 396)] * weight_shared[(((((int)threadIdx.x) % 3) + 324) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 397)] * weight_shared[(((((int)threadIdx.x) % 3) + 327) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 428)] * weight_shared[(((((int)threadIdx.x) % 3) + 228) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 429)] * weight_shared[(((((int)threadIdx.x) % 3) + 231) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 14)] * weight_shared[(((((int)threadIdx.x) % 3) + 1482) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 15)] * weight_shared[(((((int)threadIdx.x) % 3) + 1485) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 46)] * weight_shared[(((((int)threadIdx.x) % 3) + 1386) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 47)] * weight_shared[(((((int)threadIdx.x) % 3) + 1389) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 206)] * weight_shared[(((((int)threadIdx.x) % 3) + 714) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 207)] * weight_shared[(((((int)threadIdx.x) % 3) + 717) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 238)] * weight_shared[(((((int)threadIdx.x) % 3) + 618) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 239)] * weight_shared[(((((int)threadIdx.x) % 3) + 621) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 206)] * weight_shared[(((((int)threadIdx.x) % 3) + 1098) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 207)] * weight_shared[(((((int)threadIdx.x) % 3) + 1101) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 238)] * weight_shared[(((((int)threadIdx.x) % 3) + 1002) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 239)] * weight_shared[(((((int)threadIdx.x) % 3) + 1005) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 398)] * weight_shared[(((((int)threadIdx.x) % 3) + 330) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 399)] * weight_shared[(((((int)threadIdx.x) % 3) + 333) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 430)] * weight_shared[(((((int)threadIdx.x) % 3) + 234) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 431)] * weight_shared[(((((int)threadIdx.x) % 3) + 237) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 16)] * weight_shared[(((((int)threadIdx.x) % 3) + 1488) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 17)] * weight_shared[(((((int)threadIdx.x) % 3) + 1491) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 48)] * weight_shared[(((((int)threadIdx.x) % 3) + 1392) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 49)] * weight_shared[(((((int)threadIdx.x) % 3) + 1395) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 208)] * weight_shared[(((((int)threadIdx.x) % 3) + 720) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 209)] * weight_shared[(((((int)threadIdx.x) % 3) + 723) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 240)] * weight_shared[(((((int)threadIdx.x) % 3) + 624) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 241)] * weight_shared[(((((int)threadIdx.x) % 3) + 627) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 208)] * weight_shared[(((((int)threadIdx.x) % 3) + 1104) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 209)] * weight_shared[(((((int)threadIdx.x) % 3) + 1107) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 240)] * weight_shared[(((((int)threadIdx.x) % 3) + 1008) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 241)] * weight_shared[(((((int)threadIdx.x) % 3) + 1011) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 400)] * weight_shared[(((((int)threadIdx.x) % 3) + 336) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 401)] * weight_shared[(((((int)threadIdx.x) % 3) + 339) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 432)] * weight_shared[(((((int)threadIdx.x) % 3) + 240) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 433)] * weight_shared[(((((int)threadIdx.x) % 3) + 243) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 18)] * weight_shared[(((((int)threadIdx.x) % 3) + 1494) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 19)] * weight_shared[(((((int)threadIdx.x) % 3) + 1497) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 50)] * weight_shared[(((((int)threadIdx.x) % 3) + 1398) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 51)] * weight_shared[(((((int)threadIdx.x) % 3) + 1401) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 210)] * weight_shared[(((((int)threadIdx.x) % 3) + 726) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 211)] * weight_shared[(((((int)threadIdx.x) % 3) + 729) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 242)] * weight_shared[(((((int)threadIdx.x) % 3) + 630) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 243)] * weight_shared[(((((int)threadIdx.x) % 3) + 633) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 210)] * weight_shared[(((((int)threadIdx.x) % 3) + 1110) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 211)] * weight_shared[(((((int)threadIdx.x) % 3) + 1113) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 242)] * weight_shared[(((((int)threadIdx.x) % 3) + 1014) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 243)] * weight_shared[(((((int)threadIdx.x) % 3) + 1017) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 402)] * weight_shared[(((((int)threadIdx.x) % 3) + 342) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 403)] * weight_shared[(((((int)threadIdx.x) % 3) + 345) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 434)] * weight_shared[(((((int)threadIdx.x) % 3) + 246) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 435)] * weight_shared[(((((int)threadIdx.x) % 3) + 249) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 20)] * weight_shared[(((((int)threadIdx.x) % 3) + 1500) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 21)] * weight_shared[(((((int)threadIdx.x) % 3) + 1503) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 52)] * weight_shared[(((((int)threadIdx.x) % 3) + 1404) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 53)] * weight_shared[(((((int)threadIdx.x) % 3) + 1407) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 212)] * weight_shared[(((((int)threadIdx.x) % 3) + 732) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 213)] * weight_shared[(((((int)threadIdx.x) % 3) + 735) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 244)] * weight_shared[(((((int)threadIdx.x) % 3) + 636) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 245)] * weight_shared[(((((int)threadIdx.x) % 3) + 639) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 212)] * weight_shared[(((((int)threadIdx.x) % 3) + 1116) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 213)] * weight_shared[(((((int)threadIdx.x) % 3) + 1119) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 244)] * weight_shared[(((((int)threadIdx.x) % 3) + 1020) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 245)] * weight_shared[(((((int)threadIdx.x) % 3) + 1023) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 404)] * weight_shared[(((((int)threadIdx.x) % 3) + 348) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 405)] * weight_shared[(((((int)threadIdx.x) % 3) + 351) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 436)] * weight_shared[(((((int)threadIdx.x) % 3) + 252) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 437)] * weight_shared[(((((int)threadIdx.x) % 3) + 255) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 22)] * weight_shared[(((((int)threadIdx.x) % 3) + 1506) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 23)] * weight_shared[(((((int)threadIdx.x) % 3) + 1509) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 54)] * weight_shared[(((((int)threadIdx.x) % 3) + 1410) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 55)] * weight_shared[(((((int)threadIdx.x) % 3) + 1413) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 214)] * weight_shared[(((((int)threadIdx.x) % 3) + 738) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 215)] * weight_shared[(((((int)threadIdx.x) % 3) + 741) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 246)] * weight_shared[(((((int)threadIdx.x) % 3) + 642) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 247)] * weight_shared[(((((int)threadIdx.x) % 3) + 645) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 214)] * weight_shared[(((((int)threadIdx.x) % 3) + 1122) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 215)] * weight_shared[(((((int)threadIdx.x) % 3) + 1125) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 246)] * weight_shared[(((((int)threadIdx.x) % 3) + 1026) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 247)] * weight_shared[(((((int)threadIdx.x) % 3) + 1029) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 406)] * weight_shared[(((((int)threadIdx.x) % 3) + 354) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 407)] * weight_shared[(((((int)threadIdx.x) % 3) + 357) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 438)] * weight_shared[(((((int)threadIdx.x) % 3) + 258) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 439)] * weight_shared[(((((int)threadIdx.x) % 3) + 261) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 24)] * weight_shared[(((((int)threadIdx.x) % 3) + 1512) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 25)] * weight_shared[(((((int)threadIdx.x) % 3) + 1515) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 56)] * weight_shared[(((((int)threadIdx.x) % 3) + 1416) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 57)] * weight_shared[(((((int)threadIdx.x) % 3) + 1419) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 216)] * weight_shared[(((((int)threadIdx.x) % 3) + 744) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 217)] * weight_shared[(((((int)threadIdx.x) % 3) + 747) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 248)] * weight_shared[(((((int)threadIdx.x) % 3) + 648) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 249)] * weight_shared[(((((int)threadIdx.x) % 3) + 651) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 216)] * weight_shared[(((((int)threadIdx.x) % 3) + 1128) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 217)] * weight_shared[(((((int)threadIdx.x) % 3) + 1131) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 248)] * weight_shared[(((((int)threadIdx.x) % 3) + 1032) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 249)] * weight_shared[(((((int)threadIdx.x) % 3) + 1035) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 408)] * weight_shared[(((((int)threadIdx.x) % 3) + 360) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 409)] * weight_shared[(((((int)threadIdx.x) % 3) + 363) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 440)] * weight_shared[(((((int)threadIdx.x) % 3) + 264) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 441)] * weight_shared[(((((int)threadIdx.x) % 3) + 267) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 26)] * weight_shared[(((((int)threadIdx.x) % 3) + 1518) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 27)] * weight_shared[(((((int)threadIdx.x) % 3) + 1521) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 58)] * weight_shared[(((((int)threadIdx.x) % 3) + 1422) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 59)] * weight_shared[(((((int)threadIdx.x) % 3) + 1425) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 218)] * weight_shared[(((((int)threadIdx.x) % 3) + 750) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 219)] * weight_shared[(((((int)threadIdx.x) % 3) + 753) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 250)] * weight_shared[(((((int)threadIdx.x) % 3) + 654) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 251)] * weight_shared[(((((int)threadIdx.x) % 3) + 657) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 218)] * weight_shared[(((((int)threadIdx.x) % 3) + 1134) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 219)] * weight_shared[(((((int)threadIdx.x) % 3) + 1137) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 250)] * weight_shared[(((((int)threadIdx.x) % 3) + 1038) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 251)] * weight_shared[(((((int)threadIdx.x) % 3) + 1041) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 410)] * weight_shared[(((((int)threadIdx.x) % 3) + 366) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 411)] * weight_shared[(((((int)threadIdx.x) % 3) + 369) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 442)] * weight_shared[(((((int)threadIdx.x) % 3) + 270) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 443)] * weight_shared[(((((int)threadIdx.x) % 3) + 273) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 28)] * weight_shared[(((((int)threadIdx.x) % 3) + 1524) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 29)] * weight_shared[(((((int)threadIdx.x) % 3) + 1527) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 60)] * weight_shared[(((((int)threadIdx.x) % 3) + 1428) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 61)] * weight_shared[(((((int)threadIdx.x) % 3) + 1431) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 220)] * weight_shared[(((((int)threadIdx.x) % 3) + 756) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 221)] * weight_shared[(((((int)threadIdx.x) % 3) + 759) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 252)] * weight_shared[(((((int)threadIdx.x) % 3) + 660) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 253)] * weight_shared[(((((int)threadIdx.x) % 3) + 663) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 220)] * weight_shared[(((((int)threadIdx.x) % 3) + 1140) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 221)] * weight_shared[(((((int)threadIdx.x) % 3) + 1143) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 252)] * weight_shared[(((((int)threadIdx.x) % 3) + 1044) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 253)] * weight_shared[(((((int)threadIdx.x) % 3) + 1047) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 412)] * weight_shared[(((((int)threadIdx.x) % 3) + 372) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 413)] * weight_shared[(((((int)threadIdx.x) % 3) + 375) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 444)] * weight_shared[(((((int)threadIdx.x) % 3) + 276) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 445)] * weight_shared[(((((int)threadIdx.x) % 3) + 279) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 30)] * weight_shared[(((((int)threadIdx.x) % 3) + 1530) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 31)] * weight_shared[(((((int)threadIdx.x) % 3) + 1533) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 62)] * weight_shared[(((((int)threadIdx.x) % 3) + 1434) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 63)] * weight_shared[(((((int)threadIdx.x) % 3) + 1437) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 222)] * weight_shared[(((((int)threadIdx.x) % 3) + 762) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 223)] * weight_shared[(((((int)threadIdx.x) % 3) + 765) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 254)] * weight_shared[(((((int)threadIdx.x) % 3) + 666) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 255)] * weight_shared[(((((int)threadIdx.x) % 3) + 669) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 222)] * weight_shared[(((((int)threadIdx.x) % 3) + 1146) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 223)] * weight_shared[(((((int)threadIdx.x) % 3) + 1149) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 254)] * weight_shared[(((((int)threadIdx.x) % 3) + 1050) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 255)] * weight_shared[(((((int)threadIdx.x) % 3) + 1053) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 414)] * weight_shared[(((((int)threadIdx.x) % 3) + 378) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 415)] * weight_shared[(((((int)threadIdx.x) % 3) + 381) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 446)] * weight_shared[(((((int)threadIdx.x) % 3) + 282) - (rw_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1 * 32)) + 447)] * weight_shared[(((((int)threadIdx.x) % 3) + 285) - (rw_1 * 192))]));
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int rw_1_1 = 0; rw_1_1 < 2; ++rw_1_1) {
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1152)] * weight_shared[(((((int)threadIdx.x) % 3) + 2976) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1153)] * weight_shared[(((((int)threadIdx.x) % 3) + 2979) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1184)] * weight_shared[(((((int)threadIdx.x) % 3) + 2880) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1185)] * weight_shared[(((((int)threadIdx.x) % 3) + 2883) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1344)] * weight_shared[(((((int)threadIdx.x) % 3) + 2208) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1345)] * weight_shared[(((((int)threadIdx.x) % 3) + 2211) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1376)] * weight_shared[(((((int)threadIdx.x) % 3) + 2112) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1377)] * weight_shared[(((((int)threadIdx.x) % 3) + 2115) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1344)] * weight_shared[(((((int)threadIdx.x) % 3) + 2592) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1345)] * weight_shared[(((((int)threadIdx.x) % 3) + 2595) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1376)] * weight_shared[(((((int)threadIdx.x) % 3) + 2496) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1377)] * weight_shared[(((((int)threadIdx.x) % 3) + 2499) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1536)] * weight_shared[(((((int)threadIdx.x) % 3) + 1824) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1537)] * weight_shared[(((((int)threadIdx.x) % 3) + 1827) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1568)] * weight_shared[(((((int)threadIdx.x) % 3) + 1728) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1569)] * weight_shared[(((((int)threadIdx.x) % 3) + 1731) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1154)] * weight_shared[(((((int)threadIdx.x) % 3) + 2982) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1155)] * weight_shared[(((((int)threadIdx.x) % 3) + 2985) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1186)] * weight_shared[(((((int)threadIdx.x) % 3) + 2886) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1187)] * weight_shared[(((((int)threadIdx.x) % 3) + 2889) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1346)] * weight_shared[(((((int)threadIdx.x) % 3) + 2214) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1347)] * weight_shared[(((((int)threadIdx.x) % 3) + 2217) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1378)] * weight_shared[(((((int)threadIdx.x) % 3) + 2118) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1379)] * weight_shared[(((((int)threadIdx.x) % 3) + 2121) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1346)] * weight_shared[(((((int)threadIdx.x) % 3) + 2598) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1347)] * weight_shared[(((((int)threadIdx.x) % 3) + 2601) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1378)] * weight_shared[(((((int)threadIdx.x) % 3) + 2502) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1379)] * weight_shared[(((((int)threadIdx.x) % 3) + 2505) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1538)] * weight_shared[(((((int)threadIdx.x) % 3) + 1830) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1539)] * weight_shared[(((((int)threadIdx.x) % 3) + 1833) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1570)] * weight_shared[(((((int)threadIdx.x) % 3) + 1734) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1571)] * weight_shared[(((((int)threadIdx.x) % 3) + 1737) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1156)] * weight_shared[(((((int)threadIdx.x) % 3) + 2988) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1157)] * weight_shared[(((((int)threadIdx.x) % 3) + 2991) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1188)] * weight_shared[(((((int)threadIdx.x) % 3) + 2892) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1189)] * weight_shared[(((((int)threadIdx.x) % 3) + 2895) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1348)] * weight_shared[(((((int)threadIdx.x) % 3) + 2220) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1349)] * weight_shared[(((((int)threadIdx.x) % 3) + 2223) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1380)] * weight_shared[(((((int)threadIdx.x) % 3) + 2124) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1381)] * weight_shared[(((((int)threadIdx.x) % 3) + 2127) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1348)] * weight_shared[(((((int)threadIdx.x) % 3) + 2604) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1349)] * weight_shared[(((((int)threadIdx.x) % 3) + 2607) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1380)] * weight_shared[(((((int)threadIdx.x) % 3) + 2508) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1381)] * weight_shared[(((((int)threadIdx.x) % 3) + 2511) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1540)] * weight_shared[(((((int)threadIdx.x) % 3) + 1836) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1541)] * weight_shared[(((((int)threadIdx.x) % 3) + 1839) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1572)] * weight_shared[(((((int)threadIdx.x) % 3) + 1740) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1573)] * weight_shared[(((((int)threadIdx.x) % 3) + 1743) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1158)] * weight_shared[(((((int)threadIdx.x) % 3) + 2994) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1159)] * weight_shared[(((((int)threadIdx.x) % 3) + 2997) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1190)] * weight_shared[(((((int)threadIdx.x) % 3) + 2898) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1191)] * weight_shared[(((((int)threadIdx.x) % 3) + 2901) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1350)] * weight_shared[(((((int)threadIdx.x) % 3) + 2226) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1351)] * weight_shared[(((((int)threadIdx.x) % 3) + 2229) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1382)] * weight_shared[(((((int)threadIdx.x) % 3) + 2130) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1383)] * weight_shared[(((((int)threadIdx.x) % 3) + 2133) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1350)] * weight_shared[(((((int)threadIdx.x) % 3) + 2610) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1351)] * weight_shared[(((((int)threadIdx.x) % 3) + 2613) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1382)] * weight_shared[(((((int)threadIdx.x) % 3) + 2514) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1383)] * weight_shared[(((((int)threadIdx.x) % 3) + 2517) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1542)] * weight_shared[(((((int)threadIdx.x) % 3) + 1842) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1543)] * weight_shared[(((((int)threadIdx.x) % 3) + 1845) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1574)] * weight_shared[(((((int)threadIdx.x) % 3) + 1746) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1575)] * weight_shared[(((((int)threadIdx.x) % 3) + 1749) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1160)] * weight_shared[(((((int)threadIdx.x) % 3) + 3000) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1161)] * weight_shared[(((((int)threadIdx.x) % 3) + 3003) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1192)] * weight_shared[(((((int)threadIdx.x) % 3) + 2904) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1193)] * weight_shared[(((((int)threadIdx.x) % 3) + 2907) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1352)] * weight_shared[(((((int)threadIdx.x) % 3) + 2232) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1353)] * weight_shared[(((((int)threadIdx.x) % 3) + 2235) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1384)] * weight_shared[(((((int)threadIdx.x) % 3) + 2136) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1385)] * weight_shared[(((((int)threadIdx.x) % 3) + 2139) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1352)] * weight_shared[(((((int)threadIdx.x) % 3) + 2616) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1353)] * weight_shared[(((((int)threadIdx.x) % 3) + 2619) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1384)] * weight_shared[(((((int)threadIdx.x) % 3) + 2520) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1385)] * weight_shared[(((((int)threadIdx.x) % 3) + 2523) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1544)] * weight_shared[(((((int)threadIdx.x) % 3) + 1848) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1545)] * weight_shared[(((((int)threadIdx.x) % 3) + 1851) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1576)] * weight_shared[(((((int)threadIdx.x) % 3) + 1752) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1577)] * weight_shared[(((((int)threadIdx.x) % 3) + 1755) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1162)] * weight_shared[(((((int)threadIdx.x) % 3) + 3006) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1163)] * weight_shared[(((((int)threadIdx.x) % 3) + 3009) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1194)] * weight_shared[(((((int)threadIdx.x) % 3) + 2910) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1195)] * weight_shared[(((((int)threadIdx.x) % 3) + 2913) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1354)] * weight_shared[(((((int)threadIdx.x) % 3) + 2238) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1355)] * weight_shared[(((((int)threadIdx.x) % 3) + 2241) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1386)] * weight_shared[(((((int)threadIdx.x) % 3) + 2142) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1387)] * weight_shared[(((((int)threadIdx.x) % 3) + 2145) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1354)] * weight_shared[(((((int)threadIdx.x) % 3) + 2622) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1355)] * weight_shared[(((((int)threadIdx.x) % 3) + 2625) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1386)] * weight_shared[(((((int)threadIdx.x) % 3) + 2526) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1387)] * weight_shared[(((((int)threadIdx.x) % 3) + 2529) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1546)] * weight_shared[(((((int)threadIdx.x) % 3) + 1854) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1547)] * weight_shared[(((((int)threadIdx.x) % 3) + 1857) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1578)] * weight_shared[(((((int)threadIdx.x) % 3) + 1758) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1579)] * weight_shared[(((((int)threadIdx.x) % 3) + 1761) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1164)] * weight_shared[(((((int)threadIdx.x) % 3) + 3012) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1165)] * weight_shared[(((((int)threadIdx.x) % 3) + 3015) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1196)] * weight_shared[(((((int)threadIdx.x) % 3) + 2916) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1197)] * weight_shared[(((((int)threadIdx.x) % 3) + 2919) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1356)] * weight_shared[(((((int)threadIdx.x) % 3) + 2244) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1357)] * weight_shared[(((((int)threadIdx.x) % 3) + 2247) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1388)] * weight_shared[(((((int)threadIdx.x) % 3) + 2148) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1389)] * weight_shared[(((((int)threadIdx.x) % 3) + 2151) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1356)] * weight_shared[(((((int)threadIdx.x) % 3) + 2628) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1357)] * weight_shared[(((((int)threadIdx.x) % 3) + 2631) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1388)] * weight_shared[(((((int)threadIdx.x) % 3) + 2532) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1389)] * weight_shared[(((((int)threadIdx.x) % 3) + 2535) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1548)] * weight_shared[(((((int)threadIdx.x) % 3) + 1860) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1549)] * weight_shared[(((((int)threadIdx.x) % 3) + 1863) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1580)] * weight_shared[(((((int)threadIdx.x) % 3) + 1764) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1581)] * weight_shared[(((((int)threadIdx.x) % 3) + 1767) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1166)] * weight_shared[(((((int)threadIdx.x) % 3) + 3018) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1167)] * weight_shared[(((((int)threadIdx.x) % 3) + 3021) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1198)] * weight_shared[(((((int)threadIdx.x) % 3) + 2922) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1199)] * weight_shared[(((((int)threadIdx.x) % 3) + 2925) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1358)] * weight_shared[(((((int)threadIdx.x) % 3) + 2250) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1359)] * weight_shared[(((((int)threadIdx.x) % 3) + 2253) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1390)] * weight_shared[(((((int)threadIdx.x) % 3) + 2154) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1391)] * weight_shared[(((((int)threadIdx.x) % 3) + 2157) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1358)] * weight_shared[(((((int)threadIdx.x) % 3) + 2634) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1359)] * weight_shared[(((((int)threadIdx.x) % 3) + 2637) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1390)] * weight_shared[(((((int)threadIdx.x) % 3) + 2538) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1391)] * weight_shared[(((((int)threadIdx.x) % 3) + 2541) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1550)] * weight_shared[(((((int)threadIdx.x) % 3) + 1866) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1551)] * weight_shared[(((((int)threadIdx.x) % 3) + 1869) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1582)] * weight_shared[(((((int)threadIdx.x) % 3) + 1770) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1583)] * weight_shared[(((((int)threadIdx.x) % 3) + 1773) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1168)] * weight_shared[(((((int)threadIdx.x) % 3) + 3024) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1169)] * weight_shared[(((((int)threadIdx.x) % 3) + 3027) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1200)] * weight_shared[(((((int)threadIdx.x) % 3) + 2928) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1201)] * weight_shared[(((((int)threadIdx.x) % 3) + 2931) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1360)] * weight_shared[(((((int)threadIdx.x) % 3) + 2256) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1361)] * weight_shared[(((((int)threadIdx.x) % 3) + 2259) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1392)] * weight_shared[(((((int)threadIdx.x) % 3) + 2160) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1393)] * weight_shared[(((((int)threadIdx.x) % 3) + 2163) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1360)] * weight_shared[(((((int)threadIdx.x) % 3) + 2640) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1361)] * weight_shared[(((((int)threadIdx.x) % 3) + 2643) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1392)] * weight_shared[(((((int)threadIdx.x) % 3) + 2544) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1393)] * weight_shared[(((((int)threadIdx.x) % 3) + 2547) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1552)] * weight_shared[(((((int)threadIdx.x) % 3) + 1872) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1553)] * weight_shared[(((((int)threadIdx.x) % 3) + 1875) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1584)] * weight_shared[(((((int)threadIdx.x) % 3) + 1776) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1585)] * weight_shared[(((((int)threadIdx.x) % 3) + 1779) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1170)] * weight_shared[(((((int)threadIdx.x) % 3) + 3030) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1171)] * weight_shared[(((((int)threadIdx.x) % 3) + 3033) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1202)] * weight_shared[(((((int)threadIdx.x) % 3) + 2934) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1203)] * weight_shared[(((((int)threadIdx.x) % 3) + 2937) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1362)] * weight_shared[(((((int)threadIdx.x) % 3) + 2262) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1363)] * weight_shared[(((((int)threadIdx.x) % 3) + 2265) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1394)] * weight_shared[(((((int)threadIdx.x) % 3) + 2166) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1395)] * weight_shared[(((((int)threadIdx.x) % 3) + 2169) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1362)] * weight_shared[(((((int)threadIdx.x) % 3) + 2646) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1363)] * weight_shared[(((((int)threadIdx.x) % 3) + 2649) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1394)] * weight_shared[(((((int)threadIdx.x) % 3) + 2550) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1395)] * weight_shared[(((((int)threadIdx.x) % 3) + 2553) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1554)] * weight_shared[(((((int)threadIdx.x) % 3) + 1878) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1555)] * weight_shared[(((((int)threadIdx.x) % 3) + 1881) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1586)] * weight_shared[(((((int)threadIdx.x) % 3) + 1782) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1587)] * weight_shared[(((((int)threadIdx.x) % 3) + 1785) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1172)] * weight_shared[(((((int)threadIdx.x) % 3) + 3036) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1173)] * weight_shared[(((((int)threadIdx.x) % 3) + 3039) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1204)] * weight_shared[(((((int)threadIdx.x) % 3) + 2940) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1205)] * weight_shared[(((((int)threadIdx.x) % 3) + 2943) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1364)] * weight_shared[(((((int)threadIdx.x) % 3) + 2268) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1365)] * weight_shared[(((((int)threadIdx.x) % 3) + 2271) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1396)] * weight_shared[(((((int)threadIdx.x) % 3) + 2172) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1397)] * weight_shared[(((((int)threadIdx.x) % 3) + 2175) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1364)] * weight_shared[(((((int)threadIdx.x) % 3) + 2652) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1365)] * weight_shared[(((((int)threadIdx.x) % 3) + 2655) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1396)] * weight_shared[(((((int)threadIdx.x) % 3) + 2556) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1397)] * weight_shared[(((((int)threadIdx.x) % 3) + 2559) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1556)] * weight_shared[(((((int)threadIdx.x) % 3) + 1884) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1557)] * weight_shared[(((((int)threadIdx.x) % 3) + 1887) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1588)] * weight_shared[(((((int)threadIdx.x) % 3) + 1788) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1589)] * weight_shared[(((((int)threadIdx.x) % 3) + 1791) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1174)] * weight_shared[(((((int)threadIdx.x) % 3) + 3042) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1175)] * weight_shared[(((((int)threadIdx.x) % 3) + 3045) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1206)] * weight_shared[(((((int)threadIdx.x) % 3) + 2946) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1207)] * weight_shared[(((((int)threadIdx.x) % 3) + 2949) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1366)] * weight_shared[(((((int)threadIdx.x) % 3) + 2274) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1367)] * weight_shared[(((((int)threadIdx.x) % 3) + 2277) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1398)] * weight_shared[(((((int)threadIdx.x) % 3) + 2178) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1399)] * weight_shared[(((((int)threadIdx.x) % 3) + 2181) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1366)] * weight_shared[(((((int)threadIdx.x) % 3) + 2658) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1367)] * weight_shared[(((((int)threadIdx.x) % 3) + 2661) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1398)] * weight_shared[(((((int)threadIdx.x) % 3) + 2562) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1399)] * weight_shared[(((((int)threadIdx.x) % 3) + 2565) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1558)] * weight_shared[(((((int)threadIdx.x) % 3) + 1890) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1559)] * weight_shared[(((((int)threadIdx.x) % 3) + 1893) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1590)] * weight_shared[(((((int)threadIdx.x) % 3) + 1794) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1591)] * weight_shared[(((((int)threadIdx.x) % 3) + 1797) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1176)] * weight_shared[(((((int)threadIdx.x) % 3) + 3048) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1177)] * weight_shared[(((((int)threadIdx.x) % 3) + 3051) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1208)] * weight_shared[(((((int)threadIdx.x) % 3) + 2952) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1209)] * weight_shared[(((((int)threadIdx.x) % 3) + 2955) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1368)] * weight_shared[(((((int)threadIdx.x) % 3) + 2280) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1369)] * weight_shared[(((((int)threadIdx.x) % 3) + 2283) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1400)] * weight_shared[(((((int)threadIdx.x) % 3) + 2184) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1401)] * weight_shared[(((((int)threadIdx.x) % 3) + 2187) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1368)] * weight_shared[(((((int)threadIdx.x) % 3) + 2664) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1369)] * weight_shared[(((((int)threadIdx.x) % 3) + 2667) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1400)] * weight_shared[(((((int)threadIdx.x) % 3) + 2568) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1401)] * weight_shared[(((((int)threadIdx.x) % 3) + 2571) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1560)] * weight_shared[(((((int)threadIdx.x) % 3) + 1896) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1561)] * weight_shared[(((((int)threadIdx.x) % 3) + 1899) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1592)] * weight_shared[(((((int)threadIdx.x) % 3) + 1800) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1593)] * weight_shared[(((((int)threadIdx.x) % 3) + 1803) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1178)] * weight_shared[(((((int)threadIdx.x) % 3) + 3054) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1179)] * weight_shared[(((((int)threadIdx.x) % 3) + 3057) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1210)] * weight_shared[(((((int)threadIdx.x) % 3) + 2958) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1211)] * weight_shared[(((((int)threadIdx.x) % 3) + 2961) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1370)] * weight_shared[(((((int)threadIdx.x) % 3) + 2286) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1371)] * weight_shared[(((((int)threadIdx.x) % 3) + 2289) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1402)] * weight_shared[(((((int)threadIdx.x) % 3) + 2190) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1403)] * weight_shared[(((((int)threadIdx.x) % 3) + 2193) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1370)] * weight_shared[(((((int)threadIdx.x) % 3) + 2670) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1371)] * weight_shared[(((((int)threadIdx.x) % 3) + 2673) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1402)] * weight_shared[(((((int)threadIdx.x) % 3) + 2574) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1403)] * weight_shared[(((((int)threadIdx.x) % 3) + 2577) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1562)] * weight_shared[(((((int)threadIdx.x) % 3) + 1902) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1563)] * weight_shared[(((((int)threadIdx.x) % 3) + 1905) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1594)] * weight_shared[(((((int)threadIdx.x) % 3) + 1806) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1595)] * weight_shared[(((((int)threadIdx.x) % 3) + 1809) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1180)] * weight_shared[(((((int)threadIdx.x) % 3) + 3060) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1181)] * weight_shared[(((((int)threadIdx.x) % 3) + 3063) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1212)] * weight_shared[(((((int)threadIdx.x) % 3) + 2964) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1213)] * weight_shared[(((((int)threadIdx.x) % 3) + 2967) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1372)] * weight_shared[(((((int)threadIdx.x) % 3) + 2292) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1373)] * weight_shared[(((((int)threadIdx.x) % 3) + 2295) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1404)] * weight_shared[(((((int)threadIdx.x) % 3) + 2196) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1405)] * weight_shared[(((((int)threadIdx.x) % 3) + 2199) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1372)] * weight_shared[(((((int)threadIdx.x) % 3) + 2676) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1373)] * weight_shared[(((((int)threadIdx.x) % 3) + 2679) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1404)] * weight_shared[(((((int)threadIdx.x) % 3) + 2580) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1405)] * weight_shared[(((((int)threadIdx.x) % 3) + 2583) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1564)] * weight_shared[(((((int)threadIdx.x) % 3) + 1908) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1565)] * weight_shared[(((((int)threadIdx.x) % 3) + 1911) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1596)] * weight_shared[(((((int)threadIdx.x) % 3) + 1812) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1597)] * weight_shared[(((((int)threadIdx.x) % 3) + 1815) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1182)] * weight_shared[(((((int)threadIdx.x) % 3) + 3066) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1183)] * weight_shared[(((((int)threadIdx.x) % 3) + 3069) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1214)] * weight_shared[(((((int)threadIdx.x) % 3) + 2970) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1215)] * weight_shared[(((((int)threadIdx.x) % 3) + 2973) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1374)] * weight_shared[(((((int)threadIdx.x) % 3) + 2298) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[0] = (conv2d_transpose_nhwc_local[0] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1375)] * weight_shared[(((((int)threadIdx.x) % 3) + 2301) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1406)] * weight_shared[(((((int)threadIdx.x) % 3) + 2202) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[1] = (conv2d_transpose_nhwc_local[1] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1407)] * weight_shared[(((((int)threadIdx.x) % 3) + 2205) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1374)] * weight_shared[(((((int)threadIdx.x) % 3) + 2682) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1375)] * weight_shared[(((((int)threadIdx.x) % 3) + 2685) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1406)] * weight_shared[(((((int)threadIdx.x) % 3) + 2586) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1407)] * weight_shared[(((((int)threadIdx.x) % 3) + 2589) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1566)] * weight_shared[(((((int)threadIdx.x) % 3) + 1914) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[2] = (conv2d_transpose_nhwc_local[2] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1567)] * weight_shared[(((((int)threadIdx.x) % 3) + 1917) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1598)] * weight_shared[(((((int)threadIdx.x) % 3) + 1818) - (rw_1_1 * 192))]));
    conv2d_transpose_nhwc_local[3] = (conv2d_transpose_nhwc_local[3] + (PadInput_shared[(((((((int)threadIdx.x) / 12) * 192) + (((((int)threadIdx.x) % 12) / 3) * 32)) + (rw_1_1 * 32)) + 1599)] * weight_shared[(((((int)threadIdx.x) % 3) + 1821) - (rw_1_1 * 192))]));
  }
  conv2d_transpose_nhwc[((((((((int)blockIdx.x) >> 3) * 1536) + ((((int)threadIdx.x) / 12) * 384)) + ((((int)blockIdx.x) & 7) * 24)) + (((((int)threadIdx.x) % 12) / 3) * 6)) + (((int)threadIdx.x) % 3))] = conv2d_transpose_nhwc_local[0];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 3) * 1536) + ((((int)threadIdx.x) / 12) * 384)) + ((((int)blockIdx.x) & 7) * 24)) + (((((int)threadIdx.x) % 12) / 3) * 6)) + (((int)threadIdx.x) % 3)) + 3)] = conv2d_transpose_nhwc_local[1];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 3) * 1536) + ((((int)threadIdx.x) / 12) * 384)) + ((((int)blockIdx.x) & 7) * 24)) + (((((int)threadIdx.x) % 12) / 3) * 6)) + (((int)threadIdx.x) % 3)) + 192)] = conv2d_transpose_nhwc_local[2];
  conv2d_transpose_nhwc[(((((((((int)blockIdx.x) >> 3) * 1536) + ((((int)threadIdx.x) / 12) * 384)) + ((((int)blockIdx.x) & 7) * 24)) + (((((int)threadIdx.x) % 12) / 3) * 6)) + (((int)threadIdx.x) % 3)) + 195)] = conv2d_transpose_nhwc_local[3];
}

