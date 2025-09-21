// =============================================================================
// CUDA Kernel for shape [32, 26] → Total: 832 elements
// =============================================================================

__global__ void __launch_bounds__(256)
sin(const float *__restrict__ A, float *__restrict__ T_sin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 832) {
        T_sin[idx] = sinf(A[idx]);
    }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int total_elements) {
    float *d_A, *d_C;

    cudaMalloc(&d_A, total_elements * sizeof(float));
    cudaMalloc(&d_C, total_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Block size: 256 threads per block
    dim3 blockSize(256);
    // Grid size: ceil(total / block_size)
    dim3 numBlocks((total_elements + blockSize.x - 1) / blockSize.x);

    // Launch kernel
    sin<<<numBlocks, blockSize>>>(d_A, d_C);

    cudaMemcpy(h_C, d_C, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}
