// 固定参数（来自 JSON 配置）
constexpr int PARAMS_BATCH = 50;   // 行数 (N)
constexpr int PARAMS_LEN   = 128;  // 每行长度/特征数 (C)
constexpr int INDICES_LEN  = 4;    // indices 长度 (I)

// ============================================================ //
// 核函数：gather 沿 axis=1
// output[i][k] = params[i][ indices[k] ]
// 每个线程处理一个 output 元素
// ============================================================ //
__global__ void gather(const float* params,
                       const int* indices,
                       float* output) {
  int k = threadIdx.x;                    // indices 维度 [0, 3]
  int i = blockIdx.x * blockDim.x + k;    // batch 维度 [0, 49]

  if (i >= PARAMS_BATCH || k >= INDICES_LEN) return;

  int col_index = indices[k];  // 要取的列索引

  float val = 0.0f;
  if (col_index >= 0 && col_index < PARAMS_LEN) {
    val = params[i * PARAMS_LEN + col_index];
  }

  output[i * INDICES_LEN + k] = val;
}

// ============================================================ //
// Host 函数：包含 H2D、D2H、内存管理
// 输入：host 指针
// 输出：结果写回 h_output
// ============================================================ //
extern "C" void gather_kernel(const float* h_params,
                              const int* h_indices,
                              float* h_output) {
  float *d_params;
  int *d_indices;
  float *d_output;

  size_t params_bytes  = PARAMS_BATCH * PARAMS_LEN * sizeof(float);
  size_t indices_bytes = INDICES_LEN * sizeof(int);
  size_t output_bytes  = PARAMS_BATCH * INDICES_LEN * sizeof(float);

  // 1. 分配设备内存
  cudaMalloc(&d_params, params_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  // 2. Host to Device 拷贝
  cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  // 3. 配置 kernel 启动参数
  // 每个 block 处理 INDICES_LEN=4 个输出列，用 4 个线程
  dim3 block_size(INDICES_LEN);  // x: 4 线程，对应 indices_len
  dim3 grid_size((PARAMS_BATCH + block_size.x - 1) / block_size.x);  // 足够覆盖所有行

  // 4. 启动 kernel
  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output);


  // 6. Device to Host 拷贝结果
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  // 7. 释放设备内存
  cudaFree(d_params);
  cudaFree(d_indices);
  cudaFree(d_output);
}
