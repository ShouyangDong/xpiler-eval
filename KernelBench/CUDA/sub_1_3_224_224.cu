// =============================================================================
// 2. Shape: [1, 3, 224, 224] → Total: 150,528 elements (Image input)
// =============================================================================
__global__ void __launch_bounds__(1024)
sub(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 150528) {
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" void sub_kernel(float *h_A, float *h_B, float *h_C) {
    float *d_A, *d_B, *d_C;
    const int total = 1 * 3 * 224 * 224;  // 150528

    // 分配设备内存
    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    // Host → Device
    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

    // 配置 kernel 启动参数
    dim3 blockSize(1024);
    dim3 numBlocks((total + blockSize.x - 1) / blockSize.x);  // 上取整

    // ✅ 修复：使用正确的 kernel 名字 'sub'
    sub<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    // 等待 kernel 完成（可选，但建议用于调试）
    cudaDeviceSynchronize();

    // Device → Host
    cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
