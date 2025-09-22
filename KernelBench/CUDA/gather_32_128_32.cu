// 固定参数（根据你的 JSON 配置）
constexpr int PARAMS_BATCH = 32;   // 行数
constexpr int PARAMS_LEN   = 128;  // 每行长度（原特征数）
constexpr int INDICES_LEN  = 32;   // indices 长度（新特征数）

// ============================================================ //
// 核函数：gather 沿 axis=1
// 功能：output[i][j] = params[i][indices[j]] （若不越界，否则为 0）
// ============================================================ //
__global__ void gather(const float* params,
                       const int* indices,
                       float* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // j: indices 维度
  int idy = blockIdx.y * blockDim.y + threadIdx.y;  // i: batch 维度

  if (idx >= INDICES_LEN || idy >= PARAMS_BATCH) return;

  int col_index = indices[idx];  // 要取的列索引

  float val = 0.0f;
  if (col_index >= 0 && col_index < PARAMS_LEN) {
    val = params[idy * PARAMS_LEN + col_index];
  }

  output[idy * INDICES_LEN + idx] = val;
}

// 仅用于独立测试，不要和 Python 混用！
extern "C" void gather_kernel_standalone(const float* h_params,
                                        const int* h_indices,
                                        float* h_output) {
  float *d_params;
  int *d_indices;
  float *d_output;

  // 1. 分配设备内存
  cudaMalloc(&d_params, PARAMS_BATCH * PARAMS_LEN * sizeof(float));
  cudaMalloc(&d_indices, INDICES_LEN * sizeof(int));
  cudaMalloc(&d_output, PARAMS_BATCH * INDICES_LEN * sizeof(float));

  // 2. H2D
  cudaMemcpy(d_params, h_params, PARAMS_BATCH * PARAMS_LEN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, INDICES_LEN * sizeof(int), cudaMemcpyHostToDevice);

  // 3. 启动 kernel
  dim3 block_size(16, 16);
  dim3 grid_size((INDICES_LEN + 15) / 16, (PARAMS_BATCH + 15) / 16);
  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel error: %s\n", cudaGetErrorString(err));
    return;
  }

  // 4. D2H
  cudaMemcpy(h_output, d_output, PARAMS_BATCH * INDICES_LEN * sizeof(float), cudaMemcpyDeviceToHost);

  // 5. 释放
  cudaFree(d_params);
  cudaFree(d_indices);
  cudaFree(d_output);
}
