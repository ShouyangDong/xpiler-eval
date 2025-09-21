__global__ void __launch_bounds__(64)
sub(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 64) {
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" void sub_kernel(float *h_A, float *h_B, float *h_C, int total) {
    float *d_A, *d_B, *d_C;

    // 分配设备内存
    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    // Host → Device 数据拷贝
    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

    // 配置 kernel 启动参数
    dim3 blockSize(64); // 每个block有64个线程
    dim3 numBlocks(1);  // 只需要1个block来处理这64个元素

    // 启动 kernel
    sub<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    // Device → Host 数据拷贝
    cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
