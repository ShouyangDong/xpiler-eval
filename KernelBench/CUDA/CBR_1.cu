
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(32) main_kernel(float* __restrict__ bias, float* __restrict__ bn_offset, float* __restrict__ bn_scale, float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float Conv2dOutput_local[8];
  __shared__ float PaddedInput_shared[360];
  __shared__ float kernel_shared[2304];
  Conv2dOutput_local[0] = 0.000000e+00f;
  Conv2dOutput_local[1] = 0.000000e+00f;
  Conv2dOutput_local[2] = 0.000000e+00f;
  Conv2dOutput_local[3] = 0.000000e+00f;
  Conv2dOutput_local[4] = 0.000000e+00f;
  Conv2dOutput_local[5] = 0.000000e+00f;
  Conv2dOutput_local[6] = 0.000000e+00f;
  Conv2dOutput_local[7] = 0.000000e+00f;
  for (int rc_0 = 0; rc_0 < 8; ++rc_0) {
    __syncthreads();
    float condval;
    if (((28 <= ((int)blockIdx.x)) && (1 <= ((((((int)blockIdx.x) % 28) >> 2) * 8) + (((int)threadIdx.x) >> 3))))) {
      condval = data[(((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval = 0.000000e+00f;
    }
    PaddedInput_shared[((int)threadIdx.x)] = condval;
    float condval_1;
    if ((28 <= ((int)blockIdx.x))) {
      condval_1 = data[(((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3392)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 32)] = condval_1;
    float condval_2;
    if (((1 <= (((((int)blockIdx.x) / 28) * 4) + ((((int)threadIdx.x) + 64) / 72))) && (1 <= ((((((int)blockIdx.x) % 28) >> 2) * 8) + (((((int)threadIdx.x) >> 3) + 8) % 9))))) {
      condval_2 = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 64) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((((int)threadIdx.x) >> 3) + 8) % 9) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 64)] = condval_2;
    PaddedInput_shared[(((int)threadIdx.x) + 96)] = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 96) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3456)];
    float condval_3;
    if ((1 <= ((((((int)blockIdx.x) % 28) >> 2) * 8) + (((((int)threadIdx.x) >> 3) + 7) % 9)))) {
      condval_3 = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 128) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((((int)threadIdx.x) >> 3) + 7) % 9) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 128)] = condval_3;
    PaddedInput_shared[(((int)threadIdx.x) + 160)] = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 160) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3520)];
    float condval_4;
    if ((1 <= ((((((int)blockIdx.x) % 28) >> 2) * 8) + (((((int)threadIdx.x) >> 3) + 6) % 9)))) {
      condval_4 = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 192) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((((int)threadIdx.x) >> 3) + 6) % 9) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3648)];
    } else {
      condval_4 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 192)] = condval_4;
    PaddedInput_shared[(((int)threadIdx.x) + 224)] = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 224) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3584)];
    PaddedInput_shared[(((int)threadIdx.x) + 256)] = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 256) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3328)];
    float condval_5;
    if ((1 <= ((((((int)blockIdx.x) % 28) >> 2) * 8) + (((int)threadIdx.x) >> 3)))) {
      condval_5 = data[(((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) + 10688)];
    } else {
      condval_5 = 0.000000e+00f;
    }
    PaddedInput_shared[(((int)threadIdx.x) + 288)] = condval_5;
    PaddedInput_shared[(((int)threadIdx.x) + 320)] = data[((((((((((int)blockIdx.x) / 28) * 14336) + (((((int)threadIdx.x) + 320) / 72) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_0 * 8)) + (((int)threadIdx.x) & 7)) - 3392)];
    if (((int)threadIdx.x) < 8) {
      PaddedInput_shared[(((int)threadIdx.x) + 352)] = data[((((((((int)blockIdx.x) / 28) * 14336) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (rc_0 * 8)) + ((int)threadIdx.x)) + 11200)];
    }
    for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 18; ++ax0_ax1_ax2_ax3_fused_0) {
      *(float4*)(kernel_shared + ((ax0_ax1_ax2_ax3_fused_0 * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + (((((((ax0_ax1_ax2_ax3_fused_0 >> 1) * 8192) + (rc_0 * 1024)) + ((ax0_ax1_ax2_ax3_fused_0 & 1) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    }
    __syncthreads();
    for (int ry_1 = 0; ry_1 < 3; ++ry_1) {
      for (int rx_2 = 0; rx_2 < 3; ++rx_2) {
        for (int rc_2 = 0; rc_2 < 8; ++rc_2) {
          Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2)] * kernel_shared[((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7))]));
          Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2)] * kernel_shared[(((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7)) + 8)]));
          Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2)] * kernel_shared[(((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7)) + 16)]));
          Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2)] * kernel_shared[(((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7)) + 24)]));
          Conv2dOutput_local[4] = (Conv2dOutput_local[4] + (PaddedInput_shared[(((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2) + 32)] * kernel_shared[((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7))]));
          Conv2dOutput_local[5] = (Conv2dOutput_local[5] + (PaddedInput_shared[(((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2) + 32)] * kernel_shared[(((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7)) + 8)]));
          Conv2dOutput_local[6] = (Conv2dOutput_local[6] + (PaddedInput_shared[(((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2) + 32)] * kernel_shared[(((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7)) + 16)]));
          Conv2dOutput_local[7] = (Conv2dOutput_local[7] + (PaddedInput_shared[(((((((((int)threadIdx.x) >> 4) * 144) + (ry_1 * 72)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (rx_2 * 8)) + rc_2) + 32)] * kernel_shared[(((((ry_1 * 768) + (rx_2 * 256)) + (rc_2 * 32)) + (((int)threadIdx.x) & 7)) + 24)]));
        }
      }
    }
  }
  compute[(((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7))] = max((((Conv2dOutput_local[0] + bias[(((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7))]) * bn_scale[(((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7))]) + bn_offset[(((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7))]), 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7)) + 8)] = max((((Conv2dOutput_local[1] + bias[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 8)]) * bn_scale[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 8)]) + bn_offset[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 8)]), 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7)) + 16)] = max((((Conv2dOutput_local[2] + bias[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 16)]) * bn_scale[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 16)]) + bn_offset[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 16)]), 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7)) + 24)] = max((((Conv2dOutput_local[3] + bias[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 24)]) * bn_scale[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 24)]) + bn_offset[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 24)]), 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7)) + 256)] = max((((Conv2dOutput_local[4] + bias[(((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7))]) * bn_scale[(((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7))]) + bn_offset[(((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7))]), 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7)) + 264)] = max((((Conv2dOutput_local[5] + bias[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 8)]) * bn_scale[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 8)]) + bn_offset[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 8)]), 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7)) + 272)] = max((((Conv2dOutput_local[6] + bias[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 16)]) * bn_scale[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 16)]) + bn_offset[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 16)]), 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 28) * 7168) + ((((int)threadIdx.x) >> 4) * 3584)) + (((((int)blockIdx.x) % 28) >> 2) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 128)) + ((((int)blockIdx.x) & 3) * 32)) + (((int)threadIdx.x) & 7)) + 280)] = max((((Conv2dOutput_local[7] + bias[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 24)]) * bn_scale[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 24)]) + bn_offset[((((((int)blockIdx.x) & 3) * 32) + (((int)threadIdx.x) & 7)) + 24)]), 0.000000e+00f);
}

