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
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> PadInput_reindex_shared_dyn_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> weight_reindex_shared_dyn_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  half4 condval;
  if ((2 <= ((int)blockIdx.x))) {
    condval = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 4096));
  } else {
    condval = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = condval;
  half4 condval_1;
  if ((2 <= ((int)blockIdx.x))) {
    condval_1 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 3584));
  } else {
    condval_1 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = condval_1;
  half4 condval_2;
  if ((2 <= ((int)blockIdx.x))) {
    condval_2 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 3072));
  } else {
    condval_2 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = condval_2;
  half4 condval_3;
  if ((2 <= ((int)blockIdx.x))) {
    condval_3 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 2560));
  } else {
    condval_3 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = condval_3;
  half4 condval_4;
  if ((2 <= ((int)blockIdx.x))) {
    condval_4 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 2048));
  } else {
    condval_4 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = condval_4;
  half4 condval_5;
  if ((2 <= ((int)blockIdx.x))) {
    condval_5 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 1536));
  } else {
    condval_5 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = condval_5;
  half4 condval_6;
  if ((2 <= ((int)blockIdx.x))) {
    condval_6 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 1024));
  } else {
    condval_6 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = condval_6;
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = *(half4*)(inputs + ((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1536));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3072));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + (((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 8192));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  half4 condval_7;
  if ((2 <= ((int)blockIdx.x))) {
    condval_7 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 4096));
  } else {
    condval_7 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = condval_7;
  half4 condval_8;
  if ((2 <= ((int)blockIdx.x))) {
    condval_8 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 3584));
  } else {
    condval_8 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = condval_8;
  half4 condval_9;
  if ((2 <= ((int)blockIdx.x))) {
    condval_9 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 3072));
  } else {
    condval_9 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = condval_9;
  half4 condval_10;
  if ((2 <= ((int)blockIdx.x))) {
    condval_10 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 2560));
  } else {
    condval_10 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = condval_10;
  half4 condval_11;
  if ((2 <= ((int)blockIdx.x))) {
    condval_11 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 2048));
  } else {
    condval_11 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = condval_11;
  half4 condval_12;
  if ((2 <= ((int)blockIdx.x))) {
    condval_12 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 1536));
  } else {
    condval_12 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = condval_12;
  half4 condval_13;
  if ((2 <= ((int)blockIdx.x))) {
    condval_13 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 1024));
  } else {
    condval_13 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = condval_13;
  half4 condval_14;
  if ((2 <= ((int)blockIdx.x))) {
    condval_14 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 512));
  } else {
    condval_14 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = condval_14;
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = *(half4*)(inputs + ((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1536));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3072));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3584));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 16384));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 24576));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  half4 condval_15;
  if ((2 <= ((int)blockIdx.x))) {
    condval_15 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 3584));
  } else {
    condval_15 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = condval_15;
  half4 condval_16;
  if ((2 <= ((int)blockIdx.x))) {
    condval_16 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 3072));
  } else {
    condval_16 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = condval_16;
  half4 condval_17;
  if ((2 <= ((int)blockIdx.x))) {
    condval_17 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 2560));
  } else {
    condval_17 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = condval_17;
  half4 condval_18;
  if ((2 <= ((int)blockIdx.x))) {
    condval_18 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 2048));
  } else {
    condval_18 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = condval_18;
  half4 condval_19;
  if ((2 <= ((int)blockIdx.x))) {
    condval_19 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 1536));
  } else {
    condval_19 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = condval_19;
  half4 condval_20;
  if ((2 <= ((int)blockIdx.x))) {
    condval_20 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 1024));
  } else {
    condval_20 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = condval_20;
  half4 condval_21;
  if ((2 <= ((int)blockIdx.x))) {
    condval_21 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) - 512));
  } else {
    condval_21 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = condval_21;
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1536));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3072));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3584));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 32768));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 40960));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = *(half4*)(inputs + ((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1536));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3072));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5632));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6144));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6656));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7168));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 49152));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 57344));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = *(half4*)(inputs + ((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1536));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3072));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3584));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5632));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6144));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6656));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7168));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7680));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 65536));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 73728));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 1536));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3072));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 3584));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5632));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6144));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6656));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7168));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7680));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 81920));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 90112));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5632));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6144));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6656));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7168));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  half4 condval_22;
  if ((((int)blockIdx.x) < 6)) {
    condval_22 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 8192));
  } else {
    condval_22 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = condval_22;
  half4 condval_23;
  if ((((int)blockIdx.x) < 6)) {
    condval_23 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 8704));
  } else {
    condval_23 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = condval_23;
  half4 condval_24;
  if ((((int)blockIdx.x) < 6)) {
    condval_24 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 9216));
  } else {
    condval_24 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = condval_24;
  half4 condval_25;
  if ((((int)blockIdx.x) < 6)) {
    condval_25 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 9728));
  } else {
    condval_25 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = condval_25;
  half4 condval_26;
  if ((((int)blockIdx.x) < 6)) {
    condval_26 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 10240));
  } else {
    condval_26 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = condval_26;
  half4 condval_27;
  if ((((int)blockIdx.x) < 6)) {
    condval_27 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 10752));
  } else {
    condval_27 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = condval_27;
  half4 condval_28;
  if ((((int)blockIdx.x) < 6)) {
    condval_28 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 11264));
  } else {
    condval_28 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = condval_28;
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 98304));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 106496));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5632));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6144));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6656));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7168));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7680));
  half4 condval_29;
  if ((((int)blockIdx.x) < 6)) {
    condval_29 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 8192));
  } else {
    condval_29 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = condval_29;
  half4 condval_30;
  if ((((int)blockIdx.x) < 6)) {
    condval_30 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 8704));
  } else {
    condval_30 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = condval_30;
  half4 condval_31;
  if ((((int)blockIdx.x) < 6)) {
    condval_31 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 9216));
  } else {
    condval_31 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = condval_31;
  half4 condval_32;
  if ((((int)blockIdx.x) < 6)) {
    condval_32 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 9728));
  } else {
    condval_32 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = condval_32;
  half4 condval_33;
  if ((((int)blockIdx.x) < 6)) {
    condval_33 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 10240));
  } else {
    condval_33 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = condval_33;
  half4 condval_34;
  if ((((int)blockIdx.x) < 6)) {
    condval_34 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 10752));
  } else {
    condval_34 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = condval_34;
  half4 condval_35;
  if ((((int)blockIdx.x) < 6)) {
    condval_35 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 11264));
  } else {
    condval_35 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = condval_35;
  half4 condval_36;
  if ((((int)blockIdx.x) < 6)) {
    condval_36 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 11776));
  } else {
    condval_36 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = condval_36;
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 114688));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 122880));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5120)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 5664)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6208)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 5632));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 6752)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6144));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7296)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 6656));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 7840)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7168));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8384)) = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 7680));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 8928)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  half4 condval_37;
  if ((((int)blockIdx.x) < 6)) {
    condval_37 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 8704));
  } else {
    condval_37 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 9472)) = condval_37;
  half4 condval_38;
  if ((((int)blockIdx.x) < 6)) {
    condval_38 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 9216));
  } else {
    condval_38 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10016)) = condval_38;
  half4 condval_39;
  if ((((int)blockIdx.x) < 6)) {
    condval_39 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 9728));
  } else {
    condval_39 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 10560)) = condval_39;
  half4 condval_40;
  if ((((int)blockIdx.x) < 6)) {
    condval_40 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 10240));
  } else {
    condval_40 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11104)) = condval_40;
  half4 condval_41;
  if ((((int)blockIdx.x) < 6)) {
    condval_41 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 10752));
  } else {
    condval_41 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 11648)) = condval_41;
  half4 condval_42;
  if ((((int)blockIdx.x) < 6)) {
    condval_42 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 11264));
  } else {
    condval_42 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12192)) = condval_42;
  half4 condval_43;
  if ((((int)blockIdx.x) < 6)) {
    condval_43 = *(half4*)(inputs + (((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + 11776));
  } else {
    condval_43 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 12736)) = condval_43;
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 136) + (((int)threadIdx.x) * 4)) + 13280)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 131072));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2560)) = *(uint4*)(weight + ((((((((((int)threadIdx.y) >> 1) * 4096) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) & 1) * 512)) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 139264));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5120)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5136)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5152)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5168)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5184)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[2560])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5200)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3200])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5216)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[3840])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PadInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 2176) + 5232)])), 136);
  nvcuda::wmma::load_matrix_sync(weight_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[4480])), 40);
  nvcuda::wmma::mma_sync(conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], PadInput_reindex_shared_dyn_wmma_matrix_a[0], weight_reindex_shared_dyn_wmma_matrix_b[0], conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 256)])), conv2d_capsule_nhwijc_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(conv2d_capsule_nhwijc + (((((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2))) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2048)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 4096)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(conv2d_capsule_nhwijc + ((((((((((int)blockIdx.x) >> 1) * 8192) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 6144)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
}

