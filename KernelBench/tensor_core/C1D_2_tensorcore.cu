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
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ conv1d_nlc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ conv1d_nlc, half* __restrict__ inputs, half* __restrict__ weight) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> conv1d_nlc_reindex_shared_dyn_wmma_accumulator[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> PadInput_reindex_shared_dyn_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> weight_reindex_shared_dyn_wmma_matrix_b[2];
  nvcuda::wmma::fill_fragment(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  uint4 condval;
  if ((1 < (((((int)blockIdx.y) >> 1) * 16) + ((((int)blockIdx.x) >> 2) * 8)))) {
    condval = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) - 512));
  } else {
    condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8))) = condval;
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 528)) = *(uint4*)(inputs + (((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1056)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 512));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1584)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1024));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2112)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1536));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2640)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2048));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3168)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2560));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3696)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3072));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4224)) = *(uint4*)(weight + (((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4864)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 4096));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 5504)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 8192));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6144)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 12288));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6784)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 16384));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 7424)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 20480));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8064)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 24576));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8704)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 28672));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9344)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32768));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9984)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 36864));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 10624)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 40960));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11264)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 45056));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11904)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 49152));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 12544)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 53248));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13184)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 57344));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13824)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 61440));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[16])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4224)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4864)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[32])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[48])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 5504)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6144)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[64])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[80])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6784)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 7424)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[96])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[112])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8064)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8704)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[128])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[144])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9344)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9984)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[160])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[176])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 10624)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11264)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[192])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[208])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11904)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 12544)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[224])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[240])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13184)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13824)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  uint4 condval_1;
  if ((1 <= ((((((int)blockIdx.y) >> 1) * 16) + ((((int)blockIdx.x) >> 2) * 8)) + ((((int)threadIdx.y) + 1) >> 1)))) {
    condval_1 = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) - 256));
  } else {
    condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8))) = condval_1;
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 528)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 256));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1056)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 768));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1584)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1280));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2112)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1792));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2640)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2304));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3168)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2816));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3696)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3328));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4224)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 65536));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4864)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 69632));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 5504)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 73728));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6144)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 77824));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6784)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 81920));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 7424)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 86016));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8064)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 90112));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8704)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 94208));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9344)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 98304));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9984)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 102400));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 10624)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 106496));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11264)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 110592));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11904)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 114688));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 12544)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 118784));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13184)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 122880));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13824)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 126976));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[16])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4224)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4864)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[32])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[48])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 5504)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6144)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[64])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[80])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6784)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 7424)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[96])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[112])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8064)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8704)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[128])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[144])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9344)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9984)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[160])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[176])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 10624)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11264)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[192])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[208])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11904)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 12544)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[224])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[240])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13184)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13824)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8))) = *(uint4*)(inputs + (((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 528)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 512));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1056)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1024));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1584)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1536));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2112)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2048));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2640)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2560));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3168)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3072));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3696)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3584));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4224)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 131072));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4864)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 135168));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 5504)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 139264));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6144)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 143360));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6784)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 147456));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 7424)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 151552));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8064)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 155648));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8704)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 159744));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9344)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 163840));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9984)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 167936));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 10624)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 172032));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11264)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 176128));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11904)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 180224));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 12544)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 184320));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13184)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 188416));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13824)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 192512));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[16])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4224)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4864)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[32])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[48])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 5504)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6144)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[64])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[80])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6784)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 7424)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[96])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[112])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8064)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8704)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[128])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[144])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9344)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9984)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[160])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[176])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 10624)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11264)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[192])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[208])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11904)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 12544)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[224])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[240])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13184)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13824)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8))) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 256));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 528)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 768));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1056)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1280));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1584)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1792));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2112)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2304));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2640)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2816));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3168)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3328));
  uint4 condval_2;
  if ((((((((int)blockIdx.y) >> 1) * 32) + ((((int)blockIdx.x) >> 2) * 16)) + ((int)threadIdx.y)) < 49)) {
    condval_2 = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3840));
  } else {
    condval_2 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3696)) = condval_2;
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4224)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 196608));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4864)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 200704));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 5504)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 204800));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6144)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 208896));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6784)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 212992));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 7424)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 217088));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8064)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 221184));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8704)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 225280));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9344)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 229376));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9984)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 233472));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 10624)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 237568));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11264)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 241664));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11904)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 245760));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 12544)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 249856));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13184)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 253952));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13824)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 258048));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[16])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4224)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4864)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[32])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[48])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 5504)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6144)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[64])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[80])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6784)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 7424)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[96])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[112])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8064)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8704)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[128])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[144])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9344)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9984)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[160])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[176])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 10624)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11264)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[192])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[208])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11904)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 12544)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[224])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[240])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13184)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13824)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8))) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 512));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 528)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1024));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1056)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 1536));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 1584)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2048));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2112)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2560));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 2640)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3072));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3168)) = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 3584));
  uint4 condval_3;
  if (((((((int)blockIdx.y) >> 1) * 2) + (((int)blockIdx.x) >> 2)) < 3)) {
    condval_3 = *(uint4*)(inputs + ((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 4096));
  } else {
    condval_3 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 264) + (((int)threadIdx.x) * 8)) + 3696)) = condval_3;
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4224)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 262144));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4864)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 266240));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 5504)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 270336));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6144)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 274432));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6784)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 278528));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 7424)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 282624));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8064)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 286720));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8704)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 290816));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9344)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 294912));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 9984)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 299008));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 10624)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 303104));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11264)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 307200));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 11904)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 311296));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 12544)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 315392));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13184)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 319488));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 13824)) = *(uint4*)(weight + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 323584));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[16])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4224)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 4864)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[32])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[48])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 5504)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6144)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[64])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[80])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 6784)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 7424)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[96])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[112])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8064)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 8704)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[128])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[144])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9344)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 9984)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[160])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[176])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 10624)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11264)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[192])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[208])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 11904)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 12544)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[224])), 264);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[240])), 264);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13184)])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 16) + 13824)])), 40);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[1], conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 256)])), conv1d_nlc_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half4*)(conv1d_nlc + ((((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 4))) = *(half4*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)));
  *(half4*)(conv1d_nlc + (((((((((((int)blockIdx.y) >> 1) * 8192) + ((((int)blockIdx.x) >> 2) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.y) & 1) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 3) * 4)) + 16)) = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 256));
}

