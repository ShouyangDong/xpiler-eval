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
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ conv2d_nhwc, half* __restrict__ inputs, half* __restrict__ weight) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> PadInput_reindex_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> weight_reindex_shared_dyn_wmma_matrix_b[4];
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], 0.000000e+00f);
  *(half2*)(((half*)buf_dyn_shmem) + (((int)threadIdx.x) * 2)) = *(half2*)(inputs + (((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 72)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 64));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 144)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 216)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 288)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 360)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 320));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 432)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 504)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 448));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 576)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 648)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 720)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 792)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 704));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 864)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 936)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 832));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1008)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1080)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 960));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1152)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1224)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1088));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1296)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1368)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1216));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1440)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1512)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1344));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1584)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1656)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1472));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1728)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1800)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1600));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1872)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 1944)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1728));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 2016)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 2088)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1856));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 2160)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1920));
  *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 2232)) = *(half2*)(inputs + ((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + (((int)threadIdx.x) * 2)) + 1984));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2304)) = *(half2*)(weight + ((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2384)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 128));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2464)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 256));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2544)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2624)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 512));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2704)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 640));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2784)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 768));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2864)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 896));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 2944)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1024));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3024)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1152));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3104)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1280));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3184)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1408));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3264)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1536));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3344)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1664));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3424)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1792));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3504)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1920));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3584)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2048));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3664)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2176));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3744)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2304));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3824)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2432));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3904)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2560));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 3984)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2688));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4064)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2816));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4144)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2944));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4224)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3072));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4304)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3200));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4384)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3328));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4464)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3456));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4544)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3584));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4624)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3712));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4704)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3840));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 4) * 40) + ((((int)threadIdx.x) & 15) * 2)) + 4784)) = *(half2*)(weight + (((((((int)threadIdx.x) >> 4) * 64) + ((((int)blockIdx.x) & 1) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3968));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[1152])), 72);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[1168])), 72);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2304])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2320])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[2944])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[2960])), 40);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[1184])), 72);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[1200])), 72);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3584])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[3600])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[4224])), 40);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[4240])), 40);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], PadInput_reindex_shared_dyn_wmma_matrix_a[1], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[2], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], PadInput_reindex_shared_dyn_wmma_matrix_a[2], weight_reindex_shared_dyn_wmma_matrix_b[1], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], PadInput_reindex_shared_dyn_wmma_matrix_a[3], weight_reindex_shared_dyn_wmma_matrix_b[3], conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[0])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[256])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  conv2d_nhwc[(((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15))] = ((half*)buf_dyn_shmem)[((int)threadIdx.x)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 128)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 256)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 384)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 96)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 512)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 640)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 160)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 768)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 192)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 896)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 224)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 256)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 144)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 288)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 272)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 320)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 400)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 352)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 528)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 656)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 416)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 784)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 448)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 912)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 480)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[0])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[2], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[256])), conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[3], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1024)] = ((half*)buf_dyn_shmem)[((int)threadIdx.x)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1152)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1280)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1408)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 96)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1536)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1664)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 160)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1792)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 192)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1920)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 224)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1040)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 256)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1168)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 288)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1296)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 320)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1424)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 352)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1552)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1680)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 416)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1808)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 448)];
  conv2d_nhwc[((((((((int)blockIdx.y) * 28672) + ((((int)blockIdx.x) >> 1) * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 15)) + 1936)] = ((half*)buf_dyn_shmem)[(((int)threadIdx.x) + 480)];
}

