// 固定参数（来自 JSON 配置）
constexpr int PARAMS_ROWS  = 128;  // params.shape[0]
constexpr int PARAMS_COLS  = 512;  // params.shape[1]
constexpr int INDICES_LEN  = 3;    // len(indices)

// ============================================================ //
// 核函数：gather 沿 axis=0
// output[i][j] = params[indices[i]][j] （若不越界）
// ============================================================ //
__global__ void gather(const float* params,
                       const int* indices,
                       float* output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;      // 列索引 [0, 511]
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;  // 输出行 [0, 2]

  if (col >= PARAMS_COLS || out_row >= INDICES_LEN) return;

  int src_row = indices[out_row];  // 要取的源行号

  float val = 0.0f;
  if (src_row >= 0 && src_row < PARAMS_ROWS) {
    val = params[src_row * PARAMS_COLS + col];
  }

  output[out_row * PARAMS_COLS + col] = val;
}

// ============================================================ //
// Host 函数：独立版本（包含 H2D、D2H、内存管理）
// 输入：host 指针
// 输出：结果写回 host 指针 h_output
// ============================================================ //
extern "C" void gather_kernel(const float* h_params,
                              const int* h_indices,
                              float* h_output) {
  float *d_params;
  int *d_indices;
  float *d_output;

  size_t params_bytes = PARAMS_ROWS * PARAMS_COLS * sizeof(float);
  size_t indices_bytes = INDICES_LEN * sizeof(int);
  size_t output_bytes = INDICES_LEN * PARAMS_COLS * sizeof(float);

  // 1. 分配设备内存
  cudaMalloc(&d_params, params_bytes);
  cudaMalloc(&d_indices, indices_bytes);
  cudaMalloc(&d_output, output_bytes);

  // 2. Host to Device 拷贝
  cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

  // 3. 配置 kernel 启动参数
  dim3 block_size(32, 1);  // x: 32 列线程，y: 1（因 INDICES_LEN=3 很小）
  dim3 grid_size(
    (PARAMS_COLS + block_size.x - 1) / block_size.x,
    (INDICES_LEN + block_size.y - 1) / block_size.y
  );

  // 4. 启动 kernel
  gather<<<grid_size, block_size>>>(d_params, d_indices, d_output);

  // 6. Device to Host 拷贝结果
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  // 7. 释放设备内存（保证释放，即使出错）
  cudaFree(d_params);
  cudaFree(d_indices);
  cudaFree(d_output);
}