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
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, half* __restrict__ data, half* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(224) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, half* __restrict__ data, half* __restrict__ kernel) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_reindex_shared_dyn_wmma_accumulator[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> PaddedInput_reindex_shared_dyn_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> kernel_reindex_shared_dyn_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  uint4 condval;
  if (((2 <= ((int)blockIdx.y)) && (1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2))))) {
    condval = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) - 3648));
  } else {
    condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = condval;
  uint4 condval_1;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_1 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 3520));
  } else {
    condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = condval_1;
  uint4 condval_2;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_2 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 10688));
  } else {
    condval_2 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = condval_2;
  uint4 condval_3;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_3 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 17856));
  } else {
    condval_3 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = condval_3;
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + (((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  uint4 condval_4;
  if ((2 <= ((int)blockIdx.y))) {
    condval_4 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) - 3584));
  } else {
    condval_4 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = condval_4;
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 3584));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 10752));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 17920));
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 8192));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  uint4 condval_5;
  if ((2 <= ((int)blockIdx.y))) {
    condval_5 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) - 3520));
  } else {
    condval_5 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = condval_5;
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 3648));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 10816));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 17984));
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 16384));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  uint4 condval_6;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_6 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) - 64));
  } else {
    condval_6 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = condval_6;
  uint4 condval_7;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_7 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 7104));
  } else {
    condval_7 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = condval_7;
  uint4 condval_8;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_8 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 14272));
  } else {
    condval_8 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = condval_8;
  uint4 condval_9;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_9 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 21440));
  } else {
    condval_9 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = condval_9;
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 24576));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = *(uint4*)(data + (((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 7168));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 14336));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 21504));
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 32768));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 64));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 7232));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 14400));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 21568));
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 40960));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  uint4 condval_10;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_10 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 3520));
  } else {
    condval_10 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = condval_10;
  uint4 condval_11;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_11 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 10688));
  } else {
    condval_11 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = condval_11;
  uint4 condval_12;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_12 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 17856));
  } else {
    condval_12 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = condval_12;
  uint4 condval_13;
  if ((1 < ((((int)threadIdx.y) * 8) + ((((int)threadIdx.x) >> 3) * 2)))) {
    condval_13 = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 25024));
  } else {
    condval_13 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
  }
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = condval_13;
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 49152));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 3584));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 10752));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 17920));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 25088));
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 57344));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2560)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 3648));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4576)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 10816));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6592)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 17984));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8608)) = *(uint4*)(data + ((((((((int)blockIdx.y) >> 1) * 28672) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 25152));
  if (((int)threadIdx.y) < 4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(kernel + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 65536));
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2560)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2576)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2592)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1152) + 2608)])), 72);
  nvcuda::wmma::load_matrix_sync(kernel_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[1920])), 40);
  nvcuda::wmma::mma_sync(Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], PaddedInput_reindex_shared_dyn_wmma_matrix_a[0], kernel_reindex_shared_dyn_wmma_matrix_b[0], Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 256)])), Conv2dOutput_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  float2 __1;
    float2 __2;
      float2 __3;
        float2 __4;
          float2 __5;
          half2 v_ = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
          __5.x = (float)(v_.x);
          __5.y = (float)(v_.y);
          float2 v__1 = *(float2*)(bias + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
          __4.x = (__5.x+v__1.x);
          __4.y = (__5.y+v__1.y);
        float2 v__2 = *(float2*)(bn_scale + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
        __3.x = (__4.x*v__2.x);
        __3.y = (__4.y*v__2.y);
      float2 v__3 = *(float2*)(bn_offset + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
      __2.x = (__3.x+v__3.x);
      __2.y = (__3.y+v__3.y);
    float2 v__4 = make_float2(0.000000e+00f, 0.000000e+00f);
    __1.x = max(__2.x, v__4.x);
    __1.y = max(__2.y, v__4.y);
  *(float2*)(compute + (((((((((int)blockIdx.y) >> 1) * 14336) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2))) = __1;
  float2 __6;
    float2 __7;
      float2 __8;
        float2 __9;
          float2 __10;
          half2 v__5 = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 448));
          __10.x = (float)(v__5.x);
          __10.y = (float)(v__5.y);
          float2 v__6 = *(float2*)(bias + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
          __9.x = (__10.x+v__6.x);
          __9.y = (__10.y+v__6.y);
        float2 v__7 = *(float2*)(bn_scale + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
        __8.x = (__9.x*v__7.x);
        __8.y = (__9.y*v__7.y);
      float2 v__8 = *(float2*)(bn_offset + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
      __7.x = (__8.x+v__8.x);
      __7.y = (__8.y+v__8.y);
    float2 v__9 = make_float2(0.000000e+00f, 0.000000e+00f);
    __6.x = max(__7.x, v__9.x);
    __6.y = max(__7.y, v__9.y);
  *(float2*)(compute + ((((((((((int)blockIdx.y) >> 1) * 14336) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 3584)) = __6;
  float2 __11;
    float2 __12;
      float2 __13;
        float2 __14;
          float2 __15;
          half2 v__10 = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
          __15.x = (float)(v__10.x);
          __15.y = (float)(v__10.y);
          float2 v__11 = *(float2*)(bias + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
          __14.x = (__15.x+v__11.x);
          __14.y = (__15.y+v__11.y);
        float2 v__12 = *(float2*)(bn_scale + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
        __13.x = (__14.x*v__12.x);
        __13.y = (__14.y*v__12.y);
      float2 v__13 = *(float2*)(bn_offset + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
      __12.x = (__13.x+v__13.x);
      __12.y = (__13.y+v__13.y);
    float2 v__14 = make_float2(0.000000e+00f, 0.000000e+00f);
    __11.x = max(__12.x, v__14.x);
    __11.y = max(__12.y, v__14.y);
  *(float2*)(compute + ((((((((((int)blockIdx.y) >> 1) * 14336) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 7168)) = __11;
  float2 __16;
    float2 __17;
      float2 __18;
        float2 __19;
          float2 __20;
          half2 v__15 = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1344));
          __20.x = (float)(v__15.x);
          __20.y = (float)(v__15.y);
          float2 v__16 = *(float2*)(bias + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
          __19.x = (__20.x+v__16.x);
          __19.y = (__20.y+v__16.y);
        float2 v__17 = *(float2*)(bn_scale + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
        __18.x = (__19.x*v__17.x);
        __18.y = (__19.y*v__17.y);
      float2 v__18 = *(float2*)(bn_offset + ((((((int)blockIdx.y) & 1) * 64) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
      __17.x = (__18.x+v__18.x);
      __17.y = (__18.y+v__18.y);
    float2 v__19 = make_float2(0.000000e+00f, 0.000000e+00f);
    __16.x = max(__17.x, v__19.x);
    __16.y = max(__17.y, v__19.y);
  *(float2*)(compute + ((((((((((int)blockIdx.y) >> 1) * 14336) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 10752)) = __16;
}

