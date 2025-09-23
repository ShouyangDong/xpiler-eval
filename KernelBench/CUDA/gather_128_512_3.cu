// ==================== 静态维度（编译时已知）====================
constexpr int DIM0 = 128;    // params.shape[0]
constexpr int DIM1 = 512;    // params.shape[1]
constexpr int DIM2 = 3;      // params.shape[2]
constexpr int SLICE_SIZE = DIM1 * DIM2;

// ============================================================ //
// Device Kernel: 每个线程处理一个输出元素
// ============================================================ //
__global__ void gather(const float* params,
                              const int64_t* indices,
                              float* output,
                              int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * SLICE_SIZE;
    if (tid >= total_elements) return;

    int n = tid / SLICE_SIZE;           // 第 n 个 slice
    int offset_in_slice = tid % SLICE_SIZE;

    int64_t src_idx = indices[n];
    float val = 0.0f;
    if (src_idx >= 0 && src_idx < DIM0) {
        val = params[src_idx * SLICE_SIZE + offset_in_slice];
    }
    output[tid] = val;
}

// ============================================================ //
// extern "C" wrapper: 接收 host 指针，管理 device 内存
// ============================================================ //
extern "C" void gather_kernel(const float* h_params,      // host: [128, 512, 3]
                              const int64_t* h_indices,    // host: [N]，注意是 int64_t
                              float* h_output,             // host: [N, 512, 3]
                              int N) {                     // indices 长度（动态）

    // 1. 计算内存大小
    size_t params_bytes = DIM0 * DIM1 * DIM2 * sizeof(float);
    size_t indices_bytes = N * sizeof(int64_t);
    size_t output_bytes = N * DIM1 * DIM2 * sizeof(float);

    // 2. 设备指针
    float *d_params;
    int64_t *d_indices;
    float *d_output;

    // 3. 分配设备内存
    cudaMalloc(&d_params, params_bytes);
    cudaMalloc(&d_indices, indices_bytes);
    cudaMalloc(&d_output, output_bytes);

    // 4. H2D 拷贝
    cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

    // 5. 启动 kernel
    const int block_size = 256;
    int total_elements = N * DIM1 * DIM2;
    int grid_size = (total_elements + block_size - 1) / block_size;

    gather<<<grid_size, block_size>>>(d_params, d_indices, d_output, N);

    // 7. D2H 拷贝结果
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // 8. 释放设备内存
    cudaFree(d_params);
    cudaFree(d_indices);
    cudaFree(d_output);
}
