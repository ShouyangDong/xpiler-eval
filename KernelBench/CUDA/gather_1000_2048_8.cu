#include <cuda_runtime.h>
#include <stdio.h>

// 固定参数（根据你的说明）
constexpr int DIM0        = 1000;  // 第一维大小
constexpr int DIM1        = 2048;  // 第二维大小
constexpr int PARAMS_DIM2 = 8;     // 第三维大小（你明确说“输入就是1000,2048,8”）
constexpr int INDICES_LEN = 8;     // indices 长度（来自 args[2]）

// ============================================================ //
// 核函数：gather 沿 axis=2
// output[i][j][k] = params[i][j][ indices[k] ]
// 每个线程处理一个 (i, j, k) 输出元素
// ============================================================ //
__global__ void gather(const float* params,
                       const int* indices,
                       float* output) {
  int k = threadIdx.x;                    // indices 维度 [0, 7]
  int j = blockIdx.x * blockDim.x + k;    // DIM1 维度 [0, 2047]
  int i = blockIdx.y;                     // DIM0 维度 [0, 999]

  if (i >= DIM0 || j >= DIM1 || k >= INDICES_LEN) return;

  int feat_idx = indices[k];  // 要取的特征索引

  float val = 0.0f;
  if (feat_idx >= 0 && feat_idx < PARAMS_DIM2) {
    val = params[i * DIM1 * PARAMS_DIM2 + j * PARAMS_DIM2 + feat_idx];
  }

  output[i * DIM1 * INDICES_LEN + j * INDICES_LEN + k] = val;
}

// ============================================================ //
// Host 函数：包含 H2D、D2H、内存管理
// ============================================================ //
extern "C" void gather_kernel(const float* h_params,
                              const int* h_indices,
                              float* h_output) {
  float *d_params;
  int *d_indices;
  float *d_output;

  size_t params_bytes  = DIM0 * DIM1 * PARAMS_DIM2 * sizeof(float);
  size_t indices_bytes = INDICES_LEN * sizeof(int);
  size_t output_bytes  = DIM0 * DIM1 * INDICES_LEN * sizeof(float);

  // 1. 分配设备内存
  cudaMalloc(&d_params, params_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  // 2. Host to Device 拷贝
  cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  // 3. 配置 kernel 启动参数
  dim3 block_size(INDICES_LEN, 32);  // x: 8 threads (indices), y: 32 并行处理 j
  dim3 grid_size(
    (DIM1 + 31) / 32,  // 向上取整覆盖所有 j
    DIM0               // 每个 i 一个 block_y
  );

  // 4. 启动 kernel
  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output);

  // 5. 错误检查
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  cudaDeviceSynchronize();

  // 6. Device to Host 拷贝结果
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

cleanup:
  // 7. 释放设备内存
  cudaFree(d_params);
  cudaFree(d_indices);
  cudaFree(d_output);
}