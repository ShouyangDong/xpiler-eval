// ==================== 静态维度定义 ====================
constexpr int D0 = 1000;   // params.shape[0]
constexpr int D1 = 2048;   // params.shape[1]
constexpr int D2 = 8;      // params.shape[2] (axis=2)
constexpr int TOTAL_PARAMS = D0 * D1 * D2;

// ============================================================ //
// Device Kernel: 沿 axis=2 gather
// 每个线程处理 output 的一个元素: output[i][j][n]
// ============================================================ //
__global__ void gather(const float* params,
                              const int64_t* indices,
                              float* output,
                              int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = D0 * D1 * N;
    if (tid >= total_elements) return;

    // 计算输出位置
    int i = tid / (D1 * N);
    int rem = tid % (D1 * N);
    int j = rem / N;
    int n = rem % N;

    int64_t src_idx = indices[n];  // 取第 n 个索引

    float val = 0.0f;
    if (src_idx >= 0 && src_idx < D2) {  // 越界检查
        val = params[i * D1 * D2 + j * D2 + src_idx];
    }

    output[tid] = val;
}

// ============================================================ //
// extern "C" wrapper: 包含 H2D 和 D2H 拷贝
// ============================================================ //
extern "C" void gather_kernel(const float* h_params,      // host: [1000, 2048, 8]
                   const int64_t* h_indices,    // host: [N]
                   float* h_output,             // host: [1000, 2048, N]
                   int N) {

    size_t params_bytes = D0 * D1 * D2 * sizeof(float);
    size_t indices_bytes = N * sizeof(int64_t);
    size_t output_bytes = D0 * D1 * N * sizeof(float);

    float *d_params;
    int64_t *d_indices;
    float *d_output;

    cudaMalloc(&d_params, params_bytes);
    cudaMalloc(&d_indices, indices_bytes);
    cudaMalloc(&d_output, output_bytes);

    cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    int total_threads = D0 * D1 * N;
    int grid_size = (total_threads + block_size - 1) / block_size;

    gather<<<grid_size, block_size>>>(d_params, d_indices, d_output, N);

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_params);
    cudaFree(d_indices);
    cudaFree(d_output);

}
