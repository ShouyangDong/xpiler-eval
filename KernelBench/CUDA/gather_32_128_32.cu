// gather_axis1_32_128_32.cu
#include <cuda_runtime.h>
#include <stdio.h>

// ==================== 静态维度定义 ====================
constexpr int D0 = 32;     // params.shape[0]
constexpr int D1 = 128;    // params.shape[1] (axis=1)
constexpr int D2 = 32;     // params.shape[2]
constexpr int TOTAL_PARAMS = D0 * D1 * D2;

// ============================================================ //
// Device Kernel: 沿 axis=1 gather
// 每个线程处理 output 的一个元素: output[i][n][k]
// ============================================================ //
__global__ void gather_kernel(const float* params,
                              const int64_t* indices,
                              float* output,
                              int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = D0 * N * D2;
    if (tid >= total_elements) return;

    // 计算输出位置: output[i][n][k]
    int i = tid / (N * D2);
    int rem = tid % (N * D2);
    int n = rem / D2;
    int k = rem % D2;

    int64_t src_idx = indices[n];  // 取第 n 个索引

    float val = 0.0f;
    if (src_idx >= 0 && src_idx < D1) {  // 检查 axis=1 越界
        val = params[i * D1 * D2 + src_idx * D2 + k];
    }

    output[tid] = val;
}

// ============================================================ //
// extern "C" wrapper: 包含 H2D 和 D2H 拷贝
// ============================================================ //
extern "C" void gather_kernel(const float* h_params,      // host: [32, 128, 32]
                   const int64_t* h_indices,    // host: [N]
                   float* h_output,             // host: [32, N, 32]
                   int N) {

    size_t params_bytes = D0 * D1 * D2 * sizeof(float);
    size_t indices_bytes = N * sizeof(int64_t);
    size_t output_bytes = D0 * N * D2 * sizeof(float);

    float *d_params;
    int64_t *d_indices;
    float *d_output;

    cudaMalloc(&d_params, params_bytes);
    cudaMalloc(&d_indices, indices_bytes);
    cudaMalloc(&d_output, output_bytes);

    cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    int total_threads = D0 * N * D2;
    int grid_size = (total_threads + block_size - 1) / block_size;

    gather_kernel<<<grid_size, block_size>>>(d_params, d_indices, d_output, N);

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_params);
    cudaFree(d_indices);
    cudaFree(d_output);
}