// =============================================================================
// CUDA Kernel for shape [32, 64] → Total: 2,048 elements
// =============================================================================

__global__ void __launch_bounds__(1024)sin_dev(const float *__restrict__ A, float *__restrict__ T_sin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2048) {
        T_sin[idx] = sinf(A[idx]);
    }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int total_elements) {
    float *d_A, *d_C;

    cudaMalloc(&d_A, total_elements * sizeof(float));
    cudaMalloc(&d_C, total_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 使用 256 线程每 block（常见且高效）
    dim3 blockSize(1024);
    // 计算所需 block 数量：ceil(2048 / 256) = 8
    dim3 numBlocks((total_elements + blockSize.x - 1) / blockSize.x);

    // 启动 kernel
    sin_dev<<<numBlocks, blockSize>>>(d_A, d_C);

    cudaMemcpy(h_C, d_C, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}
