#include <cuda_runtime.h>
#include <iostream>

// =============================================================================
// Kernel: sub for shape [2, 16, 1024] → Total: 32,768 elements
// =============================================================================
__global__ void __launch_bounds__(256)
sub_2x16x1024(const float* __restrict__ A,
              const float* __restrict__ B,
              float* __restrict__ C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 32768) {
        C[idx] = A[idx] - B[idx];
    }
}

// =============================================================================
// Host wrapper function
// =============================================================================
extern "C" void sub_kernel_2x16x1024(float* h_A, float* h_B, float* h_C) {
    float *d_A, *d_B, *d_C;
    const int total = 2 * 16 * 1024;  // 32768

    // 分配 GPU 内存
    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    // Host → Device
    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

    // 配置 launch 参数
    dim3 blockSize(256);                    // 每 block 256 个线程
    dim3 numBlocks((total + 255) / 256);    // 上取整: (32768 + 255) / 256 = 128

    // 启动 kernel
    sub_2x16x1024<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    // 同步并检查错误
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Device → Host
    cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}