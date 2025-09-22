
// 固定参数
constexpr int DIM0        = 512;  // batch
constexpr int DIM1        = 64;   // seq_len
constexpr int PARAMS_DIM2 = 64;   // 原始特征维度（K），未在 args 中给出，假设为 64
constexpr int INDICES_LEN = 5;    // indices 长度

// ============================================================ //
// 核函数：gather 沿 axis=2
// output[i][j][k] = params[i][j][ indices[k] ]
// ============================================================ //
__global__ void gather(const float* params,
                       const int* indices,
                       float* output) {
  int k = threadIdx.x;                    // indices 维度 [0, 4]
  int j = blockIdx.x * blockDim.x + k;    // DIM1 索引（seq）
  int i = blockIdx.y;                     // DIM0 索引（batch）

  if (i >= DIM0 || j >= DIM1 || k >= INDICES_LEN) return;

  int idx = indices[k];  // 要取的特征索引

  float val = 0.0f;
  if (idx >= 0 && idx < PARAMS_DIM2) {
    val = params[i * DIM1 * PARAMS_DIM2 + j * PARAMS_DIM2 + idx];
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

  size_t params_bytes = DIM0 * DIM1 * PARAMS_DIM2 * sizeof(float);
  size_t indices_bytes = INDICES_LEN * sizeof(int);
  size_t output_bytes = DIM0 * DIM1 * INDICES_LEN * sizeof(float);

  // 1. 分配设备内存
  cudaMalloc(&d_params, params_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  // 2. H2D 拷贝
  cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  // 3. 启动 kernel
  dim3 block_size(INDICES_LEN, 32);  // x: indices_len, y: 并行处理 DIM1
  dim3 grid_size(
    (DIM1 + block_size.y - 1) / block_size.y,
    DIM0
  );

  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output);

  // 5. D2H 拷贝结果
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  // 6. 释放内存
  cudaFree(d_params);
  cudaFree(d_indices);
  cudaFree(d_output);
}