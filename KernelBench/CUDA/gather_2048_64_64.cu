// 固定参数（来自 JSON 配置）
constexpr int PARAMS_ROWS  = 2048;  // 原始行数
constexpr int PARAMS_COLS  = 64;    // 每行列数
constexpr int INDICES_LEN  = 64;    // indices 长度，也是输出行数

// ============================================================ //
// 核函数：gather 沿 axis=0
// output[k][j] = params[ indices[k] ][j]
// 每个线程处理 output[k][j]
// ============================================================ //
__global__ void gather(const float* params,
                       const int* indices,
                       float* output) {
  int j = threadIdx.x;                    // 列索引 [0, 63]
  int k = blockIdx.x * blockDim.x + j;    // 输出行索引 [0, 63]

  if (k >= INDICES_LEN || j >= PARAMS_COLS) return;

  int src_row = indices[k];  // 要取的源行号

  float val = 0.0f;
  if (src_row >= 0 && src_row < PARAMS_ROWS) {
    val = params[src_row * PARAMS_COLS + j];
  }

  output[k * PARAMS_COLS + j] = val;
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

  size_t params_bytes  = PARAMS_ROWS * PARAMS_COLS * sizeof(float);
  size_t indices_bytes = INDICES_LEN * sizeof(int);
  size_t output_bytes  = INDICES_LEN * PARAMS_COLS * sizeof(float);

  // 1. 分配设备内存
  cudaMalloc(&d_params, params_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  // 2. Host to Device 拷贝
  cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  // 3. 配置 kernel 启动参数
  // 每个 block 有 64 个线程（对应 PARAMS_COLS），处理一整行输出
  dim3 block_size(PARAMS_COLS);  // x: 64 线程，正好处理一整行输出
  dim3 grid_size(INDICES_LEN);   // 64 个 block，每个处理一个输出行

  // 4. 启动 kernel
  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output);
  // 6. Device to Host 拷贝结果
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  // 7. 释放设备内存
  cudaFree(d_params);
  cudaFree(d_indices);
  cudaFree(d_output);
}