// ==================== 静态维度定义 ====================
constexpr int D0 = 100;    // params.shape[0] (axis=0)
constexpr int D1 = 32;     // params.shape[1]
constexpr int D2 = 16;     // params.shape[2]
constexpr int TOTAL_PARAMS = D0 * D1 * D2;

// ============================================================ //
// Device Kernel: 沿 axis=0 gather
// 每个线程处理 output 的一个元素: output[n][i][j]
// ============================================================ //
__global__ void gather(const float* params,
                              const int64_t* indices,
                              float* output,
                              int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D1 * D2;
    if (tid >= total_elements) return;

    // 计算输出位置
    int n = tid / (D1 * D2);
    int rem = tid % (D1 * D2);
    int i = rem / D2;
    int j = rem % D2;

    int64_t src_idx = indices[n];

    float val = 0.0f;
    if (src_idx >= 0 && src_idx < D0) {
        val = params[src_idx * D1 * D2 + i * D2 + j];
    }

    output[tid] = val;
}

// ============================================================ //
// extern "C" wrapper: 包含 H2D 和 D2H 拷贝
// ============================================================ //
extern "C" void gather_kernel(const float* h_params,      // host: [100, 32, 16]
                   const int64_t* h_indices,    // host: [N]
                   float* h_output,             // host: [N, 32, 16]
                   int N) {

    size_t params_bytes = D0 * D1 * D2 * sizeof(float);
    size_t indices_bytes = N * sizeof(int64_t);
    size_t output_bytes = N * D1 * D2 * sizeof(float);

    float *d_params;
    int64_t *d_indices;
    float *d_output;

    cudaMalloc(&d_params, params_bytes);
    cudaMalloc(&d_indices, indices_bytes);
    cudaMalloc(&d_output, output_bytes);

    cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    int grid_size = (N * D1 * D2 + block_size - 1) / block_size;

    gather<<<grid_size, block_size>>>(d_params, d_indices, d_output, N);

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_params);
    cudaFree(d_indices);
    cudaFree(d_output);
}
