#include "hip/hip_runtime.h"

constexpr int DIM0 = 128;
constexpr int DIM1 = 512;
constexpr int DIM2 = 3;
constexpr int SLICE_SIZE = DIM1 * DIM2;

__global__ void gather(const float *params, const int64_t *indices,
                       float *output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = N * SLICE_SIZE;
  if (tid >= total_elements)
    return;

  int n = tid / SLICE_SIZE;
  int offset_in_slice = tid % SLICE_SIZE;

  int64_t src_idx = indices[n];
  float val = 0.0f;
  if (src_idx >= 0 && src_idx < DIM0) {
    val = params[src_idx * SLICE_SIZE + offset_in_slice];
  }
  output[tid] = val;
}

extern "C" void gather_kernel(const float *h_params, const int64_t *h_indices,
                              float *h_output, int N) {

  size_t params_bytes = DIM0 * DIM1 * DIM2 * sizeof(float);
  size_t indices_bytes = N * sizeof(int64_t);
  size_t output_bytes = N * DIM1 * DIM2 * sizeof(float);

  float *d_params;
  int64_t *d_indices;
  float *d_output;

  hipMalloc(&d_params, params_bytes);
  hipMalloc(&d_indices, indices_bytes);
  hipMalloc(&d_output, output_bytes);

  hipMemcpy(d_params, h_params, params_bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_indices, h_indices, indices_bytes, hipMemcpyHostToDevice);

  const int block_size = 1024;
  int total_elements = N * DIM1 * DIM2;
  int grid_size = (total_elements + block_size - 1) / block_size;

  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output, N);

  hipMemcpy(h_output, d_output, output_bytes, hipMemcpyDeviceToHost);

  hipFree(d_params);
  hipFree(d_indices);
  hipFree(d_output);
}
