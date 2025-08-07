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
#include <mma.h>

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
extern "C" __global__ void __launch_bounds__(448) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(448) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> PadInput_reindex_pad_shared_dyn_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> weight_reindex_pad_shared_dyn_wmma_matrix_b[2];
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], 0.000000e+00f);
  for (int ax2_0_0 = 0; ax2_0_0 < 5; ++ax2_0_0) {
    __syncthreads();
    half condval;
    if (((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227)) && (3 <= (((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + ((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3))))) {
      condval = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 2025)];
    } else {
      condval = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 2304)] = condval;
    half condval_1;
    if (((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227)) && (3 <= (((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3))))) {
      condval_1 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 2025)];
    } else {
      condval_1 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 2305)] = condval_1;
    half condval_2;
    if ((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227))) {
      condval_2 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 1857)];
    } else {
      condval_2 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 3424)] = condval_2;
    half condval_3;
    if ((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227))) {
      condval_3 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 1857)];
    } else {
      condval_3 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 3425)] = condval_3;
    half condval_4;
    if ((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227))) {
      condval_4 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 1689)];
    } else {
      condval_4 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 4544)] = condval_4;
    half condval_5;
    if ((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227))) {
      condval_5 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 1689)];
    } else {
      condval_5 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 4545)] = condval_5;
    half condval_6;
    if (((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227)) && ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + ((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3)) < 59))) {
      condval_6 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 1521)];
    } else {
      condval_6 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 5664)] = condval_6;
    half condval_7;
    if (((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= (((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227)) && ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3)) < 59))) {
      condval_7 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 1521)];
    } else {
      condval_7 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 5665)] = condval_7;
    half condval_8;
    if (((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + 1) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + 1) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227)) && (3 <= (((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + ((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3))))) {
      condval_8 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 681)];
    } else {
      condval_8 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 6784)] = condval_8;
    half condval_9;
    if (((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + 1) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + 1) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227)) && (3 <= (((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3))))) {
      condval_9 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 681)];
    } else {
      condval_9 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 6785)] = condval_9;
    half condval_10;
    if ((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 140) / 112)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 140) / 112)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227))) {
      condval_10 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 513)];
    } else {
      condval_10 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 7904)] = condval_10;
    half condval_11;
    if ((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 140) / 112)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 140) / 112)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227))) {
      condval_11 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 513)];
    } else {
      condval_11 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 7905)] = condval_11;
    half condval_12;
    if ((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 168) / 112)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 168) / 112)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227))) {
      condval_12 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 345)];
    } else {
      condval_12 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 9024)] = condval_12;
    half condval_13;
    if ((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 168) / 112)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 168) / 112)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227))) {
      condval_13 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 345)];
    } else {
      condval_13 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 9025)] = condval_13;
    half condval_14;
    if (((((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) < 147) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 196) / 112)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 196) / 112)) % 112) * 2) + (((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21)) < 227)) && ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + ((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3)) < 59))) {
      condval_14 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) % 21) / 3) * 3)) + (((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) % 3)) - 177)];
    } else {
      condval_14 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 10144)] = condval_14;
    half condval_15;
    if (((((((ax2_0_0 * 16) + (((int)threadIdx.x) & 15)) < 73) && (3 <= ((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 196) / 112)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)))) && (((((((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 196) / 112)) % 112) * 2) + ((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21)) < 227)) && ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) >> 4) * 2)) + (((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3)) < 59))) {
      condval_15 = inputs[((((((((((int)blockIdx.y) * 86016) + (((int)blockIdx.x) * 2688)) + (((((ax2_0_0 * 32) + ((((int)threadIdx.x) & 15) * 2)) + 1) / 21) * 672)) + (((int)threadIdx.y) * 12)) + ((((int)threadIdx.x) >> 4) * 6)) + ((((((ax2_0_0 * 11) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 21) / 3) * 3)) + ((((ax2_0_0 * 2) + ((((int)threadIdx.x) & 15) * 2)) + 1) % 3)) - 177)];
    } else {
      condval_15 = __float2half_rn(0.000000e+00f);
    }
    ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 10145)] = condval_15;
    half4 condval_16;
    if (((((ax2_0_0 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) < 147)) {
      condval_16 = *(half4*)(weight + (((ax2_0_0 * 2048) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
    } else {
      condval_16 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
    }
    *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = condval_16;
    if (((int)threadIdx.y) < 2) {
      half4 condval_17;
      if (((((ax2_0_0 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) < 119)) {
        condval_17 = *(half4*)(weight + ((((ax2_0_0 * 2048) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1792));
      } else {
        condval_17 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
      }
      *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 2016)) = condval_17;
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(PadInput_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1280) + 2304)])), 40);
    nvcuda::wmma::load_matrix_sync(PadInput_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1280) + 2944)])), 40);
    nvcuda::wmma::load_matrix_sync(weight_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) & 1) * 32)])), 72);
    nvcuda::wmma::load_matrix_sync(weight_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 16)])), 72);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[0], weight_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[0], weight_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[1], weight_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[1], weight_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::load_matrix_sync(PadInput_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1280) + 2320)])), 40);
    nvcuda::wmma::load_matrix_sync(PadInput_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1280) + 2960)])), 40);
    nvcuda::wmma::load_matrix_sync(weight_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 1152)])), 72);
    nvcuda::wmma::load_matrix_sync(weight_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 1168)])), 72);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[0], weight_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[0], weight_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[1], weight_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], PadInput_reindex_pad_shared_dyn_wmma_matrix_a[1], weight_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3]);
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 512) + 2304)])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 512) + 2560)])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half4*)(conv2d_nhwc + (((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + ((((int)threadIdx.y) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.y) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2304));
  *(half4*)(conv2d_nhwc + (((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + (((((int)threadIdx.y) + 14) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + ((((((int)threadIdx.y) >> 1) + 3) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(conv2d_nhwc + (((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + (((((int)threadIdx.y) + 28) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + ((((((int)threadIdx.y) >> 1) + 2) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 5888));
  *(half4*)(conv2d_nhwc + (((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + (((((int)threadIdx.y) + 42) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + ((((((int)threadIdx.y) >> 1) + 1) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 7680));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 512) + 2304)])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 512) + 2560)])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half4*)(conv2d_nhwc + ((((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + ((((int)threadIdx.y) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.y) & 7) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1024)) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2304));
  *(half4*)(conv2d_nhwc + ((((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + (((((int)threadIdx.y) + 14) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + ((((((int)threadIdx.y) >> 1) + 3) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1024)) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(conv2d_nhwc + ((((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + (((((int)threadIdx.y) + 28) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + ((((((int)threadIdx.y) >> 1) + 2) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1024)) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 5888));
  *(half4*)(conv2d_nhwc + ((((((((((int)blockIdx.y) * 458752) + (((int)blockIdx.x) * 14336)) + (((((int)threadIdx.y) + 42) >> 3) * 2048)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + ((((((int)threadIdx.y) >> 1) + 1) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1024)) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 7680));
}

