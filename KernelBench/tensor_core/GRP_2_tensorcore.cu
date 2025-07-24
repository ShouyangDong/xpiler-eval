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
extern "C" __global__ void __launch_bounds__(224) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(224) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight) {
  half conv2d_nhwc_local[4];
  __shared__ half PadInput_shared[1152];
  __shared__ half weight_shared[18432];
  conv2d_nhwc_local[0] = __float2half_rn(0.000000e+00f);
  conv2d_nhwc_local[1] = __float2half_rn(0.000000e+00f);
  conv2d_nhwc_local[2] = __float2half_rn(0.000000e+00f);
  conv2d_nhwc_local[3] = __float2half_rn(0.000000e+00f);
  for (int rc_0 = 0; rc_0 < 2; ++rc_0) {
    __syncthreads();
    half4 condval;
    if ((((1 <= (((((int)blockIdx.x) / 28) * 7) + (((int)threadIdx.x) >> 5))) && (1 <= ((((((int)blockIdx.x) % 28) >> 2) * 2) + ((((int)threadIdx.x) & 31) >> 3)))) && (((((((int)blockIdx.x) % 28) >> 2) * 2) + ((((int)threadIdx.x) & 31) >> 3)) < 15))) {
      condval = *(half4*)(inputs + (((((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + (rc_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) - 3840));
    } else {
      condval = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
    }
    *(half4*)(PadInput_shared + (((int)threadIdx.x) * 4)) = condval;
    if (((int)threadIdx.x) < 64) {
      half4 condval_1;
      if (((((((((int)blockIdx.x) / 28) * 7) + (((int)threadIdx.x) >> 5)) < 8) && (1 <= ((((((int)blockIdx.x) % 28) >> 2) * 2) + ((((int)threadIdx.x) & 31) >> 3)))) && (((((((int)blockIdx.x) % 28) >> 2) * 2) + ((((int)threadIdx.x) & 31) >> 3)) < 15))) {
        condval_1 = *(half4*)(inputs + (((((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 31) >> 3) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + (rc_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 21248));
      } else {
        condval_1 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
      }
      *(half4*)(PadInput_shared + ((((int)threadIdx.x) * 4) + 896)) = condval_1;
    }
    *(half4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(half4*)(weight + ((((rc_0 * 8192) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 896)) = *(half4*)(weight + (((((rc_0 * 8192) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3584));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(half4*)(weight + (((((((((int)threadIdx.x) + 448) >> 9) * 16384) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 4) + 28) & 31) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2688)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 672) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2560));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(half4*)(weight + (((((((((int)threadIdx.x) + 896) >> 9) * 16384) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 4) + 24) & 31) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4480)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 1120) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1536));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5376)) = *(half4*)(weight + (((((((((int)threadIdx.x) + 1344) >> 9) * 16384) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 4) + 20) & 31) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6272)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 1568) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 512));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7168)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 1792) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4096));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8064)) = *(half4*)(weight + (((((((((int)threadIdx.x) + 2016) >> 9) * 16384) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 4) + 30) & 31) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8960)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 2240) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3072));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 9856)) = *(half4*)(weight + (((((((((int)threadIdx.x) + 2464) >> 9) * 16384) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 4) + 26) & 31) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 10752)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 2688) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2048));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 11648)) = *(half4*)(weight + (((((((((int)threadIdx.x) + 2912) >> 9) * 16384) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 4) + 22) & 31) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 12544)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 3136) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1024));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 13440)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 3360) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4608));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 14336)) = *(half4*)(weight + (((((rc_0 * 8192) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 114688));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 15232)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 3808) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3584));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 16128)) = *(half4*)(weight + (((((((((int)threadIdx.x) + 4032) >> 9) * 16384) + (rc_0 * 8192)) + ((((((int)threadIdx.x) >> 4) + 28) & 31) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 17024)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 4256) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2560));
    if (((int)threadIdx.x) < 128) {
      *(half4*)(weight_shared + ((((int)threadIdx.x) * 4) + 17920)) = *(half4*)(weight + ((((((((((int)threadIdx.x) + 4480) >> 9) * 16384) + (rc_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6144));
    }
    __syncthreads();
    for (int rc_1 = 0; rc_1 < 2; ++rc_1) {
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16))] * weight_shared[((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2))]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[(((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16))] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 32)] * weight_shared[((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2))]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 32)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 1)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 1)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 33)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 33)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 2)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 128)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 2)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 129)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 34)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 128)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 34)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 129)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 3)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 192)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 3)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 193)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 35)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 192)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 35)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 193)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 4)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 256)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 4)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 257)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 36)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 256)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 36)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 257)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 5)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 320)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 5)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 321)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 37)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 320)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 37)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 321)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 6)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 384)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 6)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 385)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 38)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 384)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 38)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 385)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 7)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 448)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 7)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 449)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 39)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 448)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 39)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 449)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 8)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 512)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 8)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 513)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 40)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 512)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 40)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 513)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 9)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 576)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 9)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 577)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 41)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 576)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 41)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 577)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 10)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 640)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 10)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 641)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 42)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 640)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 42)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 641)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 11)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 704)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 11)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 705)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 43)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 704)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 43)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 705)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 12)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 768)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 12)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 769)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 44)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 768)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 44)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 769)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 13)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 832)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 13)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 833)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 45)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 832)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 45)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 833)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 14)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 896)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 14)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 897)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 46)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 896)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 46)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 897)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 15)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 960)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 15)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 961)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 47)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 960)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 47)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 961)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 32)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 32)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 64)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 64)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 33)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 33)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 65)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 65)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 34)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2176)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 34)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2177)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 66)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2176)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 66)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2177)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 35)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2240)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 35)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2241)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 67)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2240)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 67)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2241)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 36)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2304)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 36)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2305)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 68)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2304)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 68)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2305)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 37)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2368)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 37)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2369)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 69)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2368)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 69)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2369)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 38)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2432)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 38)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2433)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 70)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2432)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 70)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2433)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 39)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2496)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 39)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2497)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 71)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2496)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 71)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2497)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 40)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2560)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 40)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2561)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 72)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2560)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 72)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2561)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 41)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2624)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 41)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2625)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 73)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2624)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 73)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2625)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 42)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2688)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 42)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2689)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 74)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2688)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 74)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2689)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 43)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2752)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 43)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2753)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 75)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2752)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 75)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2753)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 44)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2816)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 44)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2817)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 76)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2816)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 76)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2817)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 45)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2880)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 45)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2881)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 77)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2880)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 77)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2881)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 46)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2944)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 46)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2945)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 78)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2944)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 78)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 2945)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 47)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 3008)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 47)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 3009)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 79)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 3008)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 79)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 3009)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 64)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4096)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 64)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4097)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 96)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4096)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 96)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4097)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 65)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4160)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 65)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4161)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 97)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4160)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 97)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4161)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 66)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4224)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 66)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4225)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 98)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4224)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 98)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4225)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 67)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4288)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 67)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4289)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 99)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4288)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 99)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4289)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 68)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4352)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 68)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4353)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 100)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4352)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 100)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4353)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 69)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4416)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 69)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4417)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 101)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4416)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 101)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4417)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 70)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4480)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 70)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4481)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 102)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4480)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 102)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4481)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 71)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4544)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 71)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4545)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 103)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4544)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 103)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4545)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 72)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4608)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 72)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4609)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 104)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4608)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 104)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4609)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 73)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4672)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 73)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4673)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 105)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4672)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 105)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4673)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 74)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4736)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 74)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4737)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 106)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4736)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 106)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4737)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 75)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4800)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 75)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4801)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 107)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4800)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 107)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4801)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 76)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4864)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 76)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4865)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 108)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4864)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 108)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4865)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 77)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4928)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 77)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4929)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 109)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4928)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 109)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4929)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 78)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4992)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 78)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4993)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 110)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4992)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 110)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 4993)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 79)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 5056)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 79)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 5057)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 111)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 5056)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 111)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 5057)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 128)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6144)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 128)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6145)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 160)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6144)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 160)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6145)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 129)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6208)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 129)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6209)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 161)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6208)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 161)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6209)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 130)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6272)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 130)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6273)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 162)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6272)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 162)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6273)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 131)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6336)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 131)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6337)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 163)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6336)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 163)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6337)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 132)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6400)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 132)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6401)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 164)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6400)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 164)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6401)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 133)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6464)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 133)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6465)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 165)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6464)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 165)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6465)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 134)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6528)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 134)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6529)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 166)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6528)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 166)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6529)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 135)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6592)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 135)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6593)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 167)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6592)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 167)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6593)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 136)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6656)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 136)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6657)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 168)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6656)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 168)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6657)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 137)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6720)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 137)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6721)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 169)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6720)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 169)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6721)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 138)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6784)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 138)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6785)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 170)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6784)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 170)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6785)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 139)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6848)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 139)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6849)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 171)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6848)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 171)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6849)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 140)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6912)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 140)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6913)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 172)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6912)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 172)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6913)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 141)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6976)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 141)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6977)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 173)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6976)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 173)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 6977)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 142)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7040)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 142)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7041)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 174)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7040)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 174)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7041)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 143)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7104)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 143)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7105)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 175)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7104)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 175)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 7105)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 160)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8192)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 160)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8193)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 192)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8192)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 192)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8193)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 161)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8256)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 161)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8257)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 193)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8256)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 193)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8257)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 162)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8320)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 162)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8321)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 194)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8320)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 194)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8321)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 163)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8384)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 163)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8385)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 195)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8384)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 195)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8385)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 164)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8448)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 164)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8449)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 196)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8448)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 196)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8449)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 165)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8512)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 165)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8513)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 197)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8512)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 197)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8513)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 166)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8576)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 166)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8577)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 198)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8576)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 198)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8577)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 167)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8640)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 167)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8641)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 199)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8640)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 199)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8641)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 168)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8704)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 168)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8705)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 200)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8704)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 200)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8705)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 169)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8768)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 169)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8769)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 201)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8768)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 201)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8769)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 170)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8832)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 170)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8833)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 202)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8832)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 202)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8833)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 171)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8896)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 171)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8897)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 203)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8896)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 203)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8897)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 172)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8960)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 172)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8961)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 204)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8960)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 204)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 8961)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 173)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9024)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 173)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9025)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 205)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9024)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 205)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9025)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 174)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9088)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 174)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9089)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 206)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9088)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 206)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9089)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 175)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9152)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 175)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9153)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 207)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9152)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 207)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 9153)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 192)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10240)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 192)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10241)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 224)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10240)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 224)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10241)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 193)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10304)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 193)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10305)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 225)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10304)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 225)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10305)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 194)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10368)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 194)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10369)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 226)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10368)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 226)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10369)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 195)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10432)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 195)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10433)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 227)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10432)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 227)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10433)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 196)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10496)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 196)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10497)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 228)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10496)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 228)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10497)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 197)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10560)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 197)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10561)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 229)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10560)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 229)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10561)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 198)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10624)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 198)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10625)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 230)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10624)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 230)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10625)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 199)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10688)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 199)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10689)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 231)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10688)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 231)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10689)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 200)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10752)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 200)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10753)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 232)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10752)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 232)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10753)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 201)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10816)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 201)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10817)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 233)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10816)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 233)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10817)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 202)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10880)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 202)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10881)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 234)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10880)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 234)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10881)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 203)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10944)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 203)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10945)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 235)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10944)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 235)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 10945)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 204)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11008)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 204)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11009)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 236)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11008)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 236)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11009)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 205)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11072)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 205)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11073)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 237)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11072)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 237)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11073)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 206)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11136)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 206)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11137)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 238)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11136)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 238)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11137)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 207)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11200)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 207)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11201)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 239)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11200)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 239)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 11201)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 256)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12288)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 256)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12289)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 288)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12288)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 288)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12289)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 257)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12352)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 257)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12353)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 289)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12352)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 289)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12353)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 258)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12416)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 258)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12417)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 290)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12416)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 290)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12417)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 259)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12480)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 259)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12481)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 291)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12480)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 291)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12481)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 260)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12544)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 260)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12545)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 292)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12544)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 292)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12545)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 261)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12608)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 261)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12609)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 293)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12608)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 293)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12609)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 262)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12672)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 262)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12673)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 294)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12672)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 294)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12673)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 263)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12736)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 263)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12737)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 295)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12736)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 295)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12737)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 264)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12800)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 264)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12801)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 296)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12800)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 296)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12801)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 265)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12864)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 265)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12865)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 297)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12864)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 297)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12865)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 266)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12928)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 266)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12929)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 298)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12928)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 298)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12929)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 267)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12992)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 267)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12993)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 299)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12992)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 299)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 12993)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 268)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13056)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 268)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13057)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 300)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13056)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 300)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13057)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 269)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13120)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 269)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13121)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 301)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13120)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 301)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13121)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 270)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13184)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 270)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13185)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 302)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13184)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 302)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13185)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 271)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13248)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 271)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13249)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 303)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13248)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 303)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 13249)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 288)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14336)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 288)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14337)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 320)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14336)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 320)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14337)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 289)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14400)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 289)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14401)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 321)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14400)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 321)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14401)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 290)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14464)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 290)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14465)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 322)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14464)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 322)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14465)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 291)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14528)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 291)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14529)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 323)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14528)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 323)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14529)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 292)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14592)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 292)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14593)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 324)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14592)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 324)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14593)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 293)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14656)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 293)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14657)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 325)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14656)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 325)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14657)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 294)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14720)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 294)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14721)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 326)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14720)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 326)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14721)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 295)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14784)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 295)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14785)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 327)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14784)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 327)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14785)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 296)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14848)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 296)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14849)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 328)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14848)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 328)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14849)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 297)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14912)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 297)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14913)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 329)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14912)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 329)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14913)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 298)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14976)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 298)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14977)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 330)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14976)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 330)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 14977)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 299)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15040)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 299)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15041)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 331)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15040)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 331)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15041)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 300)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15104)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 300)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15105)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 332)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15104)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 332)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15105)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 301)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15168)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 301)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15169)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 333)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15168)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 333)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15169)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 302)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15232)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 302)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15233)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 334)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15232)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 334)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15233)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 303)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15296)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 303)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15297)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 335)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15296)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 335)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 15297)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 320)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16384)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 320)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16385)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 352)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16384)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 352)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16385)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 321)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16448)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 321)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16449)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 353)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16448)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 353)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16449)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 322)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16512)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 322)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16513)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 354)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16512)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 354)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16513)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 323)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16576)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 323)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16577)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 355)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16576)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 355)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16577)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 324)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16640)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 324)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16641)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 356)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16640)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 356)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16641)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 325)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16704)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 325)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16705)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 357)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16704)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 357)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16705)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 326)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16768)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 326)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16769)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 358)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16768)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 358)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16769)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 327)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16832)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 327)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16833)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 359)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16832)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 359)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16833)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 328)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16896)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 328)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16897)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 360)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16896)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 360)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16897)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 329)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16960)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 329)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16961)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 361)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16960)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 361)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 16961)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 330)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17024)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 330)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17025)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 362)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17024)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 362)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17025)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 331)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17088)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 331)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17089)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 363)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17088)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 363)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17089)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 332)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17152)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 332)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17153)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 364)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17152)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 364)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17153)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 333)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17216)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 333)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17217)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 365)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17216)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 365)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17217)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 334)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17280)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 334)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17281)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 366)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17280)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 366)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17281)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 335)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17344)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 335)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17345)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 367)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17344)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (PadInput_shared[((((((int)threadIdx.x) >> 5) * 128) + (rc_1 * 16)) + 367)] * weight_shared[(((rc_1 * 1024) + ((((int)threadIdx.x) & 31) * 2)) + 17345)]));
    }
  }
  conv2d_nhwc[((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 31) * 2))] = conv2d_nhwc_local[0];
  conv2d_nhwc[(((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1)] = conv2d_nhwc_local[1];
  conv2d_nhwc[(((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 256)] = conv2d_nhwc_local[2];
  conv2d_nhwc[(((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) >> 5) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)blockIdx.x) & 3) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 257)] = conv2d_nhwc_local[3];
}

