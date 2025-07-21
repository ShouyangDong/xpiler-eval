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
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ conv2d_capsule_nhwijc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ conv2d_capsule_nhwijc, half* __restrict__ inputs, half* __restrict__ weight) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> PadInput_reindex_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> weight_reindex_shared_dyn_wmma_matrix_b[8];
  nvcuda::wmma::fill_fragment(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  half2 condval;
  if ((2 <= ((int)blockIdx.y))) {
    condval = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 7680));
  } else {
    condval = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = condval;
  half2 condval_1;
  if ((2 <= ((int)blockIdx.y))) {
    condval_1 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 7424));
  } else {
    condval_1 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = condval_1;
  half2 condval_2;
  if ((2 <= ((int)blockIdx.y))) {
    condval_2 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 6656));
  } else {
    condval_2 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = condval_2;
  half2 condval_3;
  if ((2 <= ((int)blockIdx.y))) {
    condval_3 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 6400));
  } else {
    condval_3 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = condval_3;
  half2 condval_4;
  if ((2 <= ((int)blockIdx.y))) {
    condval_4 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 5632));
  } else {
    condval_4 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = condval_4;
  half2 condval_5;
  if ((2 <= ((int)blockIdx.y))) {
    condval_5 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 5376));
  } else {
    condval_5 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = condval_5;
  half2 condval_6;
  if ((2 <= ((int)blockIdx.y))) {
    condval_6 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 4608));
  } else {
    condval_6 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = condval_6;
  half2 condval_7;
  if ((2 <= ((int)blockIdx.y))) {
    condval_7 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 4352));
  } else {
    condval_7 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = condval_7;
  half2 condval_8;
  if ((2 <= ((int)blockIdx.y))) {
    condval_8 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 3584));
  } else {
    condval_8 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = condval_8;
  half2 condval_9;
  if ((2 <= ((int)blockIdx.y))) {
    condval_9 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 3328));
  } else {
    condval_9 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = condval_9;
  half2 condval_10;
  if ((2 <= ((int)blockIdx.y))) {
    condval_10 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 2560));
  } else {
    condval_10 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = condval_10;
  half2 condval_11;
  if ((2 <= ((int)blockIdx.y))) {
    condval_11 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 2304));
  } else {
    condval_11 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = condval_11;
  half2 condval_12;
  if ((2 <= ((int)blockIdx.y))) {
    condval_12 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 1536));
  } else {
    condval_12 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = condval_12;
  half2 condval_13;
  if ((2 <= ((int)blockIdx.y))) {
    condval_13 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 1280));
  } else {
    condval_13 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = condval_13;
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8704));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8960));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9728));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9984));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10752));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11008));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11776));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12032));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12800));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13056));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13824));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14080));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14848));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15104));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + (((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 8192));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 8704));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 12288));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 12800));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half2 condval_14;
  if ((2 <= ((int)blockIdx.y))) {
    condval_14 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 8192));
  } else {
    condval_14 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = condval_14;
  half2 condval_15;
  if ((2 <= ((int)blockIdx.y))) {
    condval_15 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 7936));
  } else {
    condval_15 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = condval_15;
  half2 condval_16;
  if ((2 <= ((int)blockIdx.y))) {
    condval_16 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 7168));
  } else {
    condval_16 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = condval_16;
  half2 condval_17;
  if ((2 <= ((int)blockIdx.y))) {
    condval_17 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 6912));
  } else {
    condval_17 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = condval_17;
  half2 condval_18;
  if ((2 <= ((int)blockIdx.y))) {
    condval_18 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 6144));
  } else {
    condval_18 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = condval_18;
  half2 condval_19;
  if ((2 <= ((int)blockIdx.y))) {
    condval_19 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 5888));
  } else {
    condval_19 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = condval_19;
  half2 condval_20;
  if ((2 <= ((int)blockIdx.y))) {
    condval_20 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 5120));
  } else {
    condval_20 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = condval_20;
  half2 condval_21;
  if ((2 <= ((int)blockIdx.y))) {
    condval_21 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 4864));
  } else {
    condval_21 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = condval_21;
  half2 condval_22;
  if ((2 <= ((int)blockIdx.y))) {
    condval_22 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 4096));
  } else {
    condval_22 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = condval_22;
  half2 condval_23;
  if ((2 <= ((int)blockIdx.y))) {
    condval_23 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 3840));
  } else {
    condval_23 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = condval_23;
  half2 condval_24;
  if ((2 <= ((int)blockIdx.y))) {
    condval_24 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 3072));
  } else {
    condval_24 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = condval_24;
  half2 condval_25;
  if ((2 <= ((int)blockIdx.y))) {
    condval_25 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 2816));
  } else {
    condval_25 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = condval_25;
  half2 condval_26;
  if ((2 <= ((int)blockIdx.y))) {
    condval_26 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 2048));
  } else {
    condval_26 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = condval_26;
  half2 condval_27;
  if ((2 <= ((int)blockIdx.y))) {
    condval_27 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 1792));
  } else {
    condval_27 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = condval_27;
  half2 condval_28;
  if ((2 <= ((int)blockIdx.y))) {
    condval_28 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 1024));
  } else {
    condval_28 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = condval_28;
  half2 condval_29;
  if ((2 <= ((int)blockIdx.y))) {
    condval_29 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 768));
  } else {
    condval_29 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = condval_29;
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8448));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9216));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9472));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10240));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10496));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11264));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11520));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12288));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12544));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13312));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13568));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14336));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14592));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15360));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15616));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 16384));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 16896));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 20480));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 20992));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 24576));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 25088));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 28672));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 29184));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half2 condval_30;
  if ((2 <= ((int)blockIdx.y))) {
    condval_30 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 7680));
  } else {
    condval_30 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = condval_30;
  half2 condval_31;
  if ((2 <= ((int)blockIdx.y))) {
    condval_31 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 7424));
  } else {
    condval_31 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = condval_31;
  half2 condval_32;
  if ((2 <= ((int)blockIdx.y))) {
    condval_32 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 6656));
  } else {
    condval_32 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = condval_32;
  half2 condval_33;
  if ((2 <= ((int)blockIdx.y))) {
    condval_33 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 6400));
  } else {
    condval_33 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = condval_33;
  half2 condval_34;
  if ((2 <= ((int)blockIdx.y))) {
    condval_34 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 5632));
  } else {
    condval_34 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = condval_34;
  half2 condval_35;
  if ((2 <= ((int)blockIdx.y))) {
    condval_35 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 5376));
  } else {
    condval_35 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = condval_35;
  half2 condval_36;
  if ((2 <= ((int)blockIdx.y))) {
    condval_36 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 4608));
  } else {
    condval_36 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = condval_36;
  half2 condval_37;
  if ((2 <= ((int)blockIdx.y))) {
    condval_37 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 4352));
  } else {
    condval_37 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = condval_37;
  half2 condval_38;
  if ((2 <= ((int)blockIdx.y))) {
    condval_38 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 3584));
  } else {
    condval_38 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = condval_38;
  half2 condval_39;
  if ((2 <= ((int)blockIdx.y))) {
    condval_39 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 3328));
  } else {
    condval_39 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = condval_39;
  half2 condval_40;
  if ((2 <= ((int)blockIdx.y))) {
    condval_40 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 2560));
  } else {
    condval_40 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = condval_40;
  half2 condval_41;
  if ((2 <= ((int)blockIdx.y))) {
    condval_41 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 2304));
  } else {
    condval_41 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = condval_41;
  half2 condval_42;
  if ((2 <= ((int)blockIdx.y))) {
    condval_42 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 1536));
  } else {
    condval_42 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = condval_42;
  half2 condval_43;
  if ((2 <= ((int)blockIdx.y))) {
    condval_43 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 1280));
  } else {
    condval_43 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = condval_43;
  half2 condval_44;
  if ((2 <= ((int)blockIdx.y))) {
    condval_44 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 512));
  } else {
    condval_44 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = condval_44;
  half2 condval_45;
  if ((2 <= ((int)blockIdx.y))) {
    condval_45 = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) - 256));
  } else {
    condval_45 = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = condval_45;
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8704));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8960));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9728));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9984));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10752));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11008));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11776));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12032));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12800));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13056));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13824));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14080));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14848));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15104));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15872));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 16128));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 32768));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 33280));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 36864));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 37376));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 40960));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 41472));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 45056));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 45568));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 2560));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 2816));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 3584));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 3840));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 4608));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 4864));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 5632));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 5888));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 6656));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 6912));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 16896));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 17152));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 17920));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 18176));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 18944));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 19200));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 19968));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 20224));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 20992));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 21248));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 22016));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 22272));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 23040));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 23296));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 49152));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 49664));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 53248));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 53760));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 57344));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 57856));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 61440));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 61952));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = *(half2*)(inputs + ((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 2048));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 2304));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 3072));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 3328));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 4096));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 4352));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 5120));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 5376));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 6144));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 6400));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 7168));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 7424));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 16384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 16640));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 17408));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 17664));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 18432));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 18688));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 19456));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 19712));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 20480));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 20736));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 21504));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 21760));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 22528));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 22784));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 23552));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 23808));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 65536));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 66048));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 69632));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 70144));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 73728));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 74240));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 77824));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 78336));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 2560));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 2816));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 3584));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 3840));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 4608));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 4864));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 5632));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 5888));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 6656));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 6912));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 7680));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 7936));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 16896));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 17152));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 17920));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 18176));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 18944));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 19200));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 19968));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 20224));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 20992));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 21248));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 22016));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 22272));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 23040));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 23296));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 24064));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 24320));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 81920));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 82432));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 86016));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 86528));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 90112));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 90624));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 94208));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 94720));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8704));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8960));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9728));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9984));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10752));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11008));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11776));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12032));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12800));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13056));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13824));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14080));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14848));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15104));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = make_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 25088));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 25344));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 26112));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 26368));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 27136));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 27392));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 28160));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 28416));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 29184));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 29440));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 30208));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 30464));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 31232));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 31488));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 98304));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 98816));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 102400));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 102912));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 106496));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 107008));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 110592));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 111104));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8448));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9216));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9472));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10240));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10496));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11264));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11520));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12288));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12544));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13312));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13568));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14336));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14592));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15360));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15616));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 24576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 24832));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 25600));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 25856));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 26624));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 26880));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 27648));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 27904));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 28672));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 28928));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 29696));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 29952));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 30720));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 30976));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 31744));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 32000));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 114688));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 115200));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 118784));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 119296));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 122880));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 123392));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 126976));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 127488));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5120)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8704));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5392)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 8960));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5664)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9728));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 5936)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 9984));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6208)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 10752));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6480)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11008));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 6752)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 11776));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7024)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12032));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7296)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 12800));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7568)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13056));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 7840)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 13824));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8112)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14080));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8384)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 14848));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8656)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15104));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 8928)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 15872));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9200)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 16128));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9472)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 25088));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 9744)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 25344));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10016)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 26112));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10288)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 26368));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10560)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 27136));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 10832)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 27392));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11104)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 28160));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11376)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 28416));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11648)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 29184));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 11920)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 29440));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12192)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 30208));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12464)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 30464));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 12736)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 31232));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13008)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 31488));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13280)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 32256));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) >> 1) * 136) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 2)) + 13552)) = *(half2*)(inputs + (((((((int)blockIdx.y) >> 1) * 32768) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 32512));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4))) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 131072));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 131584));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1280)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 135168));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 1920)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 135680));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 2560)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 139264));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3200)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 139776));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 3840)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 143360));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 4480)) = *(half4*)(weight + ((((((((int)blockIdx.y) & 1) * 2048) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 143872));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[656])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[1296])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[1936])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2576])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[3216])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[3856])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[4496])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[4], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[6], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[5], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[7], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 512)])), conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 512) + 256)])), conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(conv2d_capsule_nhwijc + (((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2))) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 16)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 2048)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 2064)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 4096)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 4112)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 6144)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.y) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 6160)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
}

